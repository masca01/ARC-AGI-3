"""Minimal stand-in for `agents.agent.Agent` from the official ARC-AGI-3-Agents framework.

It has the same constructor, the same `main()` loop and the same attributes
(`frames`, `action_counter`, `game_id`, `MAX_ACTIONS`), so an agent written
against it drops into the official framework (and Kaggle) unchanged.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState


class Agent(ABC):
    MAX_ACTIONS: int = 80

    def __init__(
        self,
        card_id: str,
        game_id: str,
        agent_name: str,
        ROOT_URL: str,
        record: bool,
        arc_env: Any,
        tags: Optional[list[str]] = None,
    ) -> None:
        self.card_id = card_id
        self.game_id = game_id
        self.agent_name = agent_name
        self.ROOT_URL = ROOT_URL
        self.tags = tags or []
        self.arc_env = arc_env
        self.frames: list[FrameData] = [FrameData(levels_completed=0)]
        self.action_counter = 0
        self.timer = 0.0

    # ------------------------------------------------------------------ loop
    def main(self) -> None:
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            latest = self._convert(self.arc_env.observation_space)
            action = self.choose_action(self.frames, latest)
            frame = self.take_action(action)
            if frame is not None:
                self.frames.append(frame)
            self.action_counter += 1

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        data = action.action_data.model_dump()
        raw = self.arc_env.step(action, data=data)
        return None if raw is None else self._convert(raw)

    @staticmethod
    def _convert(raw: Any) -> FrameData:
        if raw is None:
            raise ValueError("environment returned no frame")
        return FrameData(
            game_id=raw.game_id,
            frame=[getattr(a, "tolist", lambda: a)() for a in raw.frame],
            state=raw.state,
            levels_completed=raw.levels_completed,
            win_levels=raw.win_levels,
            guid=raw.guid,
            full_reset=raw.full_reset,
            available_actions=raw.available_actions,
        )

    # -------------------------------------------------------------- helpers
    @property
    def state(self) -> GameState:
        return self.frames[-1].state

    @property
    def levels_completed(self) -> int:
        return self.frames[-1].levels_completed

    @property
    def seconds(self) -> float:
        return round(time.time() - self.timer, 2)

    @property
    def name(self) -> str:
        return f"{self.game_id}.{self.__class__.__name__.lower()}"

    # ------------------------------------------------------------- contract
    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool: ...

    @abstractmethod
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction: ...
