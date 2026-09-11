"""MyAgent: a neural-network-only ARC-AGI-3 agent.

Strategy (details in README.md):
  step 1  random tries        the first WARMUP_STEPS actions are random, to collect experience
  step 2  what changes?       the CNN learns which actions/clicks change the screen and lead
                              to screens it has rarely seen, and prefers those
  step 3  what wins?          when a level is completed, the moves that led to it get credit;
                              the CNN learns these patterns and reuses them on the next levels

Contract (same as the official framework and Kaggle):
  class MyAgent(Agent) with is_done(frames, latest_frame) and choose_action(frames, latest_frame)
"""
from __future__ import annotations

import random
import zlib
from typing import Any

import numpy as np
from arcengine import FrameData, GameAction, GameState

try:  # inside the official ARC-AGI-3-Agents framework (and on Kaggle)
    from agents.agent import Agent  # type: ignore
except ImportError:  # local development in this repo
    from .base import Agent

from .model import CLICK_ID, GRID, SIMPLE_IDS, SIMPLE_INDEX, Brain, to_array


class MyAgent(Agent):
    MAX_ACTIONS = 300
    WARMUP_STEPS = 20        # step 1: purely random tries
    EPSILON = 0.1            # after warm-up, still 10% random tries so the agent never gets stuck
    WIN_WEIGHT = 5.0         # how much "led to a win before" counts against "changes the screen"
    CLICK_SHARPNESS = 3.0    # > 1 concentrates clicks on the most promising pixels

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        seed = zlib.crc32(self.game_id.encode())
        random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.brain = Brain(seed=seed % 10_000)
        self.prev: tuple | None = None          # last move, waiting to see its outcome
        self.trace: list[dict] = []             # one entry per step, for replays
        self.level_steps: list[int] = []        # action count at which each level was completed
        self.last_choice: dict[str, Any] = {}
        self.static_pixels = 0
        self._hits: list[int] = []

    # ----------------------------------------------------------------- contract
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        frame = to_array(latest_frame)
        self._learn(frame, latest_frame)

        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.brain.attempt_failed()
            self.last_choice = {"action": "RESET", "p": None, "why": "start of game or game over"}
            return GameAction.RESET

        available = latest_frame.available_actions or list(range(1, 8))
        simple = [a for a in SIMPLE_IDS if a in available]
        can_click = CLICK_ID in available
        heat, heat_title, p, p_static = np.zeros((GRID, GRID)), "network not trained yet", None, None

        if self.action_counter < self.WARMUP_STEPS or not self.brain.ready or random.random() < self.EPSILON:
            # step 1: random try (always during warm-up, then EPSILON of the time)
            action = random.choice(simple + ([CLICK_ID] if can_click else []))
            x, y = (random.randrange(GRID), random.randrange(GRID)) if action == CLICK_ID else (-1, -1)
            why = "step 1 - random try"
        else:
            out = self.brain.predict(frame)
            p_static = out["static"]
            self.static_pixels = int((p_static > 0.5).sum())
            # score = P(changes the screen) + WIN_WEIGHT * P(leads to a win)
            options = {a: out["change_simple"][SIMPLE_INDEX[a]] + self.WIN_WEIGHT * out["win_simple"][SIMPLE_INDEX[a]]
                       for a in simple}
            click_map = out["change_click"] + self.WIN_WEIGHT * out["win_click"]
            if can_click:
                options[CLICK_ID] = float(click_map.max())
            acts = list(options)
            w = np.array([options[a] for a in acts], dtype=np.float64) + 1e-3
            action = int(self.rng.choice(acts, p=w / w.sum()))

            if action == CLICK_ID:
                probs = (click_map.astype(np.float64) + 1e-6) ** self.CLICK_SHARPNESS
                y, x = divmod(int(self.rng.choice(GRID * GRID, p=probs.ravel() / probs.sum())), GRID)
                p_change, p_win = out["change_click"][y, x], out["win_click"][y, x]
            else:
                x = y = -1
                p_change, p_win = out["change_simple"][SIMPLE_INDEX[action]], out["win_simple"][SIMPLE_INDEX[action]]
            p = float(p_change)
            why = ("step 3 - win pattern: similar moves led to a level win" if self.WIN_WEIGHT * p_win > p_change
                   else "step 2 - CNN predicts this leads somewhere new")
            if can_click:
                heat, heat_title = click_map / max(1e-6, float(click_map.max())), "CNN: where clicking looks useful"
            else:
                heat, heat_title = out["static"], "CNN: pixels that change anyway (counters)"

        self.prev = (frame, action, x, y, latest_frame.levels_completed, p, p_static)
        self.trace.append(dict(frame=frame, action=action, x=x, y=y, p=p, why=why,
                               level=latest_frame.levels_completed, heat=heat, heat_title=heat_title))
        name = f"ACTION6@({x},{y})" if action == CLICK_ID else f"ACTION{action}"
        self.last_choice = {"action": name, "p": p, "why": why}

        game_action = GameAction.from_id(action)
        if action == CLICK_ID:
            game_action.set_data({"x": int(x), "y": int(y)})
        return game_action

    # ----------------------------------------------------------------- learning
    def _learn(self, frame, latest_frame: FrameData) -> None:
        """Close the loop on the previous move: store it, and give win credit if a level was completed."""
        if self.prev is None:
            return
        prev_frame, action, x, y, prev_levels, p, p_static = self.prev
        self.prev = None
        target = self.brain.add(prev_frame, action, x, y, frame, p_static)
        if p is not None:  # was the CNN right about "this leads somewhere new"?
            self._hits = (self._hits + [int((p > 0.5) == (target > 0.5))])[-100:]
        if latest_frame.levels_completed > prev_levels:
            self.brain.level_completed()
            self.level_steps.append(self.action_counter)

    # ---------------------------------------------------------------- reporting
    def stats(self) -> dict[str, Any]:
        return {
            "buffer": len(self.brain.buffer),
            "screens": len(self.brain.visits),
            "train_steps": self.brain.train_steps,
            "loss": sum(self.brain.last_loss.values()) if self.brain.last_loss else None,
            "change_acc": (sum(self._hits) / len(self._hits)) if self._hits else None,
            "static_px": self.static_pixels,
            "level_steps": list(self.level_steps),
        }
