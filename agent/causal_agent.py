"""CausalAgent: the teammate's ls20 causal planner behind the official agent contract,
with optional neural perception.

  PERCEPTION = "heuristic"   original hand-made player detector (colour 12, fixed shape)
  PERCEPTION = "neural"      the CNN in control_net.py finds the player. It learns what the
                             player looks like from the agent's own moves; until it is ready the
                             agent makes a few warm-up moves to collect motion labels.

Everything else (keys, doors, tiles, gates, HUD reading, planning) is the teammate's code in
ls20_specialist.py. After a GAME_OVER the agent RESETs and retries the same level, keeping what it
learned about it.

Contract (same as the official framework and Kaggle):
  class CausalAgent(Agent) with is_done(frames, latest_frame) and choose_action(frames, latest_frame)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

try:  # inside the official ARC-AGI-3-Agents framework (and on Kaggle)
    from agents.agent import Agent  # type: ignore
except ImportError:  # local development in this repo
    from .base import Agent

from .control_net import MOVE_IDS, ControlPerception
from .ls20_specialist import LS20ExtendedStateAgent
from .model import to_array


class CausalAgent(Agent):
    MAX_ACTIONS = 1000
    PERCEPTION = "heuristic"     # or "neural"
    ORACLE = False               # neural only: also run the hand-made detector, for accuracy stats

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.planner = LS20ExtendedStateAgent(max_actions=10**9, seed=0, verbose=False)
        self.perception = ControlPerception(seed=0) if self.PERCEPTION == "neural" else None
        if self.perception is not None:
            self.planner.locate_player = self.perception.locate
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
            if latest_frame.state is GameState.GAME_OVER:
                self.planner.restart_attempt()
                self.attempts += 1
            self.prev = None
            self.last_choice = {"action": "RESET", "why": "start of game or game over"}
            return GameAction.RESET

        if self.perception is not None and self.prev is not None:
            self.perception.observe(self.prev[0], self.prev[1], frame)
        if self.perception is not None and self.ORACLE:
            self._score_cnn(frame, level)

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

        self.prev = (frame, action.value)
        self.level_actions[level] += 1
        heat = self.perception.predict(frame)[0] if self.perception is not None and self.perception.ready else None
        self.trace.append(dict(frame=frame, action=action.value, x=-1, y=-1, p=None, why=why, level=level,
                               heat=heat if heat is not None else np.zeros_like(frame, dtype=float),
                               heat_title="CNN: what I control" if heat is not None else "CNN not trained yet"))
        self.last_choice = {"action": action.name, "why": why}
        return action

    # ---------------------------------------------------------------- reporting
    def _score_cnn(self, frame: np.ndarray, level: int) -> None:
        """Compare the CNN with the hand-made detector on this frame (diagnostics only).

        Frames before the CNN passed its self-check are not scored; they are the warm-up moves.
        """
        if not self.perception.trusted:
            return
        try:
            truth = self.planner.detect_player(frame)
        except RuntimeError:
            return
        self.oracle_hits[level].append(int(self.perception.locate(frame) == truth))

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
        }
