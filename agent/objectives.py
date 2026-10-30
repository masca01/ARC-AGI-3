"""ObjectiveTracker: which goals of a level are still open, decided only from what is visible.

Game-agnostic. It knows nothing about doors or keys. Perception hands it the goal candidates it sees
each step (an id, a grid position, a pixel box, any payload), and the tracker decides their status:

  open       visible and not yet achieved
  achieved   it is no longer detected AND its pixels that are in sight and not covered by the player
             look different
             (object permanence: something only hidden behind the player still shows its
             uncovered parts unchanged; something that is gone does not)
  reopened   an achieved objective became visible again (the game put it back, for example after
             a life was lost): it is open again
  no effect  the player stood on it, left, and it was still there. After FAIL_AFTER such visits it
             is ranked behind the other open objectives, so the agent tries something else instead
             of looping on it

This fixes levels with several goals: once the first is achieved the planner forgets it and works
on the next one. In ls20 the candidates are the doors; in another game they would be whatever the
perception thinks the goals are.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np

Cell = tuple[int, int]
Box = tuple[int, int, int, int]      # y0, x0, y1, x1 (inclusive) in pixels


@dataclass
class Objective:
    id: Hashable
    pos: Cell
    payload: Any
    box: Box | None = None
    appearance: np.ndarray | None = None   # its pixels the last time it was detected
    status: str = "open"            # "open" or "achieved"
    visible: bool = True
    hidden_since: int | None = None  # step it stopped being visible
    visits: int = 0                  # times the player stood on it while it was open
    no_effect: int = 0               # visits after which it was still there
    history: list[tuple[int, str]] = field(default_factory=list)


class ObjectiveTracker:
    FAIL_AFTER = 2       # visits without effect before the objective is ranked last
    CHANGED = 0.2        # share of its uncovered pixels that must differ for "gone"

    def __init__(self) -> None:
        self.objectives: dict[Hashable, Objective] = {}
        self.events: list[tuple[str, str]] = []   # filled on every update, read by the agent for replays
        self._on: Hashable | None = None          # objective the player is standing on

    def reset_level(self) -> None:
        self.objectives = {}
        self._on = None

    def gone(self, o: Objective, frame: np.ndarray, player_box: Box | None, seen: np.ndarray | None = None) -> bool:
        """Has the objective really disappeared, or is it only covered by the player or out of sight?"""
        if o.box is None or o.appearance is None:
            return True
        y0, x0, y1, x1 = o.box
        now = frame[y0:y1 + 1, x0:x1 + 1]
        uncovered = np.ones(now.shape, bool)
        if player_box is not None:
            py0, px0, py1, px1 = player_box
            uncovered[max(0, py0 - y0):max(0, py1 - y0 + 1), max(0, px0 - x0):max(0, px1 - x0 + 1)] = False
        if seen is not None:
            uncovered &= seen[y0:y1 + 1, x0:x1 + 1]    # pixels in darkness say nothing either
        if not uncovered.any():
            return False                       # fully covered: cannot tell, wait
        return float((now[uncovered] != o.appearance[uncovered]).mean()) > self.CHANGED

    def update(self, t: int, visible: dict[Hashable, tuple[Cell, Box | None, Any]], player: Cell,
               frame: np.ndarray, player_box: Box | None, seen: np.ndarray | None = None) -> list[tuple[str, str]]:
        """Feed this step's visible goal candidates, the player's cell and box, the frame, and
        optionally which pixels can be seen at all (limited vision)."""
        self.events = []
        for oid, (pos, box, payload) in visible.items():
            o = self.objectives.get(oid)
            if box is not None:
                appearance = frame[box[0]:box[2] + 1, box[1]:box[3] + 1].copy()
            if o is None:
                self.objectives[oid] = Objective(oid, pos, payload, box, appearance if box is not None else None)
                self._event(t, self.objectives[oid], "discovered", f"objective discovered at {pos}")
                continue
            o.pos, o.payload, o.box = pos, payload, box
            if box is not None:
                o.appearance = appearance
            if o.status == "achieved":
                o.status = "open"
                self._event(t, o, "reopened", f"objective at {pos} is back: open again")
            o.visible, o.hidden_since = True, None
        for o in self.objectives.values():
            if o.id in visible:
                continue
            if o.visible:
                o.visible, o.hidden_since = False, t
            if o.status == "open" and self.gone(o, frame, player_box, seen):
                o.status = "achieved"
                self._event(t, o, "achieved", f"objective at {o.pos} achieved: gone, not just hidden")
        # visits: standing on an open objective, then leaving it while it is still there = no effect
        on = next((o.id for o in self.objectives.values() if o.pos == player and o.status == "open"), None)
        if self._on is not None and on != self._on:
            o = self.objectives.get(self._on)
            if o is not None and o.status == "open" and o.visible:
                o.no_effect += 1
                self._event(t, o, "no effect", f"stood on objective at {o.pos}, nothing happened "
                                               f"({o.no_effect}x)")
        if on is not None and on != self._on:
            self.objectives[on].visits += 1
        self._on = on
        return self.events

    def _event(self, t: int, o: Objective, kind: str, text: str) -> None:
        o.history.append((t, kind))
        self.events.append((kind, text))

    # ------------------------------------------------------------------ queries
    def is_open(self, oid: Hashable) -> bool:
        o = self.objectives.get(oid)
        return o is None or o.status == "open"

    def open(self) -> list[Objective]:
        return [o for o in self.objectives.values() if o.status == "open"]

    def penalty(self, oid: Hashable) -> int:
        """Rank key: 1 for objectives that did not react FAIL_AFTER times, else 0."""
        o = self.objectives.get(oid)
        return int(o is not None and o.no_effect >= self.FAIL_AFTER)

    def summary(self) -> str:
        return ", ".join(f"{o.pos}: {o.status}{' (no effect %dx)' % o.no_effect if o.no_effect else ''}"
                         for o in self.objectives.values()) or "-"
