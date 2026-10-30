"""CausalAgent: the teammate's ls20 causal planner behind the official agent contract,
with optional neural perception.

  PERCEPTION = "heuristic"   original hand-made player detector (colour 12, fixed shape)
  PERCEPTION = "neural"      the CNN in control_net.py finds the player. It learns what the
                             player looks like from the agent's own moves; until it is ready the
                             agent makes a few warm-up moves to collect motion labels.

  SLIDES = True              SlideNet (slide_net.py) learns from one or two surprises which icons push the
                             player, in which direction and how far, and the planner uses that everywhere.

Everything else (keys, doors, tiles, gates, HUD reading, planning) is the teammate's code in
ls20_specialist.py. After a GAME_OVER the agent RESETs and retries the same level, keeping what it
learned about it.

Contract (same as the official framework and Kaggle):
  class CausalAgent(Agent) with is_done(frames, latest_frame) and choose_action(frames, latest_frame)
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

try:  # inside the official ARC-AGI-3-Agents framework (and on Kaggle)
    from agents.agent import Agent  # type: ignore
except ImportError:  # local development in this repo
    from .base import Agent

from .control_net import MOVE_IDS, ControlPerception
from .ls20_specialist import LS20ExtendedStateAgent
from .slide_net import NAMES as SLIDE_NAMES
from .slide_net import SlideLearner
from .model import to_array


ROLE_LABELS = {"rotate_cw": "rot+90", "rotate_ccw": "rot-90", "rotate_180": "rot180", "recolor": "colour",
               "reshape": "shape", "resource": "refill", "none": "none", "unknown": "?"}


class CausalAgent(Agent):
    MAX_ACTIONS = 1000
    PERCEPTION = "heuristic"     # or "neural"
    ORACLE = False               # neural only: also run the hand-made detector, for accuracy stats
    SLIDES = False               # learn pushes with SlideNet

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.planner = LS20ExtendedStateAgent(max_actions=10**9, seed=0, verbose=False)
        self.perception = ControlPerception(seed=0) if self.PERCEPTION == "neural" else None
        if self.perception is not None:
            self.planner.locate_player = self.perception.locate
        self.slides = SlideLearner(seed=0) if self.SLIDES else None
        self.slide_events: list = []             # filled by the slide hook during this step
        if self.slides is not None:
            self.planner.slide_model = self._slide_hook
        self.prev: tuple[np.ndarray, int] | None = None   # (frame, action id) waiting for its outcome
        self.warmup_moves = 0
        self.attempts = 1
        self.level_actions = [0]                 # non-RESET actions spent on each level
        self.oracle_hits: list[list[int]] = [[]] # per level: 1 if the CNN found the player's cell
        self.trace: list[dict] = []
        self.last_choice: dict[str, Any] = {}

    # ----------------------------------------------------------------- contract
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        frame = to_array(latest_frame)
        level = latest_frame.levels_completed
        while len(self.level_actions) <= level:
            self.level_actions.append(0)
            self.oracle_hits.append([])

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            events = []
            if latest_frame.state is GameState.GAME_OVER:
                self.planner.restart_attempt()
                self.attempts += 1
                events.append(("death", f"game over: retrying level {level + 1} (attempt {self.attempts})"))
                self._record(frame, GameAction.RESET, "game over -> RESET", level, events, None, None, False)
            self.prev = None
            self.last_choice = {"action": "RESET", "why": "start of game or game over"}
            return GameAction.RESET

        motion = None
        was_trusted = self.perception is not None and self.perception.trusted
        if self.perception is not None and self.prev is not None:
            motion = self.perception.observe(self.prev[0], self.prev[1], frame)
        oracle = self._score_cnn(frame, level) if self.perception is not None and self.ORACLE else None
        events = []
        if self.trace and level > self.trace[-1]["level"]:
            events.append(("level", f"level {level} solved in {self.level_actions[level - 1]} actions"))
        if self.perception is not None and self.perception.trusted and not was_trusted:
            events.append(("cnn", "CNN passed its self-check: the planner now uses it to find the player"))
        if oracle is not None and oracle[0] != oracle[1]:
            events.append(("miss", f"CNN put the player in cell {oracle[1]}, true cell {oracle[0]}"))
        roles = {sig: m.role for sig, m in self.planner.models.items()}
        n_edges = len(self.planner.dynamic_edges)

        self.slide_events = []
        planned = True
        try:
            action = self.planner.next_action(latest_frame)
            why = self.planner.reason
        except RuntimeError as err:
            # the player could not be found: the CNN is not trained yet (or lost it). Move to
            # collect motion labels; the planner must not learn from a move it did not choose.
            self.planner.last_scene = self.planner.last_action = None
            available = [a for a in MOVE_IDS if a in (latest_frame.available_actions or MOVE_IDS)]
            action = GameAction.from_id(available[self.warmup_moves % len(available)])
            self.warmup_moves += 1
            why = f"warm-up move ({err})"
            planned = False

        if planned:
            for sig, model in self.planner.models.items():
                if model.role != roles.get(sig):
                    before = ROLE_LABELS.get(roles[sig], roles[sig]) + " -> " if sig in roles else ""
                    events.append(("learn", f"tile type {sig[:6]} does: {before}{ROLE_LABELS.get(model.role, model.role)}"))
            if len(self.planner.dynamic_edges) > n_edges:
                edge = list(self.planner.dynamic_edges.values())[-1]
                events.append(("edge", f"non-local move {edge.source} -> {edge.destination} "
                                       f"({len(self.planner.dynamic_edges)} known this level)"))
        events += self.slide_events
        self.prev = (frame, action.value)
        self.level_actions[level] += 1
        self._record(frame, action, why, level, events, motion, oracle, planned)
        self.last_choice = {"action": action.name, "why": why}
        return action

    def _slide_hook(self, previous, delta, scene):
        """Called by the planner each step: learn from the last move, then predict every cell."""
        event = self.slides.observe(previous, delta, scene) if previous is not None else None
        if event is not None and event.observed:
            name = SLIDE_NAMES[event.observed]
            if event.fast:
                self.slide_events.append(("surprise", f"surprise push {name} from {event.cell} to {event.end} "
                                                      f"(P={event.p_observed:.2f}): fast learning"))
            else:
                self.slide_events.append(("slide", f"push {name} {event.cell} -> {event.end} predicted "
                                                   f"(P={event.p_observed:.2f})"))
        elif event is not None and event.fast:
            self.slide_events.append(("surprise", f"false alarm at {event.cell}: no push (fast learning)"))
        return self.slides.table(scene)

    # ------------------------------------------------------------------ replays
    def _record(self, frame, action: GameAction, why: str, level: int, events: list, motion, oracle,
                planned: bool) -> None:
        """One replay step: what the agent saw, believed and planned (read by run_local's GIF/HTML)."""
        p, scene = self.perception, self.planner.last_scene if planned else None
        step = dict(frame=frame, action=action.value, action_name=action.name, x=-1, y=-1, p=None, why=why,
                    level=level, attempt=self.attempts, events=events, overlay={}, info={}, stats={})
        if p is not None and p.ready:
            control, by_itself = p.predict(frame)
            step["maps"] = [("CNN: what I control", control.astype(np.float16)),
                            ("CNN: changes by itself", by_itself.astype(np.float16))]
            step["layers"] = [(t, m.astype(np.float16)) for t, m in p.layer_maps(frame)]
            corner = p.top_left(frame)
            if corner is not None:
                step["overlay"]["cnn_box"] = (*corner, *p.size)
            step["stats"].update(cnn_conf=p.last_confidence, loss=p.last_loss)
        if motion is not None:
            arrived = np.roll(np.roll(motion.moved, motion.shift[0], 0), motion.shift[1], 1)
            step["overlay"]["motion"] = arrived
            step["info"]["motion label"] = f"moved by {motion.shift}, {int(motion.moved.sum())} px"
        if p is not None:
            step["info"].update({"CNN trusted": p.trusted, "CNN confident": p.confident,
                                 "labelled moves": p.n_labels, "train steps": p.train_steps,
                                 "learned grid": f"step {p.step}px, origin {p.origin}" if p.step else "-"})
        if scene is not None:
            oy, ox = scene.origin
            px = lambda cell: (oy + 5 * cell[0] + 2, ox + 5 * cell[1] + 2)   # centre of a 5x5 grid cell
            pos, config, cells = scene.player, scene.dynamic_config, [scene.player]
            for a in self.planner.last_plan:
                nxt = self.planner.transition_position(pos, a, config)
                if nxt is None:
                    break
                (pos, config) = nxt
                cells.append(pos)
            target = re.search(r"@\((\d+), (\d+)\)", why)
            step["overlay"].update(
                path=[px(c) for c in cells],
                target=px((int(target[1]), int(target[2]))) if target else (px(cells[-1]) if len(cells) > 1 else None),
                tiles=[(*px(pos), ROLE_LABELS.get(self.planner.models[t.signature].role, "?")
                        if t.signature in self.planner.models else "?") for pos, t in self.planner.tiles.items()],
                doors=[px(d.pos) for d in self.planner.doors.values()])
            if oracle is not None and oracle[0] is not None:
                (r, c), (gy, gx) = oracle[0]
                step["overlay"]["oracle_box"] = (gy + 5 * r, gx + 5 * c, 5, 5)
            if self.slides is not None:
                step["overlay"]["slides"] = [(*px(c), SLIDE_NAMES[k], round(pr, 2), *px(end))
                                             for c, (k, pr, end) in self.slides.last_table.items()
                                             if k and end is not None and pr >= 0.5]
                step["info"].update({f"slides: {k}": v for k, v in self.slides.stats().items()})
            step["info"].update({"player cell": scene.player, "plan length": len(self.planner.last_plan),
                                 "needs": self.planner.required_role(scene) or "-",
                                 "learned roles": ", ".join(sorted({ROLE_LABELS.get(m.role, m.role)
                                                                     for m in self.planner.models.values()})) or "-",
                                 "non-local moves known": len(self.planner.dynamic_edges)})
            step["key"] = list(scene.key)
            step["goals"] = [list(d.goal) for d in self.planner.doors.values()]
            step["stats"].update(resource=scene.resource, lives=scene.lives)
        maps = step.get("maps")
        step["heat"] = maps[0][1].astype(np.float32) if maps else np.zeros(frame.shape)
        step["heat_title"] = maps[0][0] if maps else ("CNN not trained yet" if p is not None else "no CNN (hand-made perception)")
        self.trace.append(step)

    # ---------------------------------------------------------------- reporting
    def _score_cnn(self, frame: np.ndarray, level: int):
        """Compare the CNN with the hand-made detector on this frame (diagnostics only).

        Returns (true placement, CNN placement). Frames before the CNN passed its self-check are not
        scored; they are the warm-up moves.
        """
        try:
            truth = self.planner.detect_player(frame)
        except RuntimeError:
            return None
        if not self.perception.trusted:
            return truth, truth
        guess = self.perception.locate(frame)
        self.oracle_hits[level].append(int(guess == truth))
        return truth, guess

    def stats(self) -> dict[str, Any]:
        p = self.perception
        hits = [h for level in self.oracle_hits for h in level]
        return {
            "level_actions": list(self.level_actions),
            "attempts": self.attempts,
            "reason": self.planner.reason,
            "warmup_moves": self.warmup_moves,
            "labels": len(p.buffer) if p else None,
            "step_px": p.step if p else None,
            "player_acc": sum(hits) / len(hits) if hits else None,
            "player_acc_by_level": [round(sum(h) / len(h), 3) if h else None for h in self.oracle_hits],
            "loss": p.last_loss if p else None,
            "models": {k: v.role for k, v in self.planner.models.items()},
            "slides": self.slides.stats() if self.slides is not None else None,
        }
