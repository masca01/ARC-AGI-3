"""GaugeFinder: discovers a resource bar from how it behaves, with no position or colour given.

Why: ls20_specialist reads the energy at a fixed place and colour (rows 61-62, columns 13-54,
colour 11), which is true for ls20 only. What makes a bar a bar is its behaviour:

  drain    on almost every action one straight slice changes from a colour c to a colour d, the slice
           touches the previous one, and together they stay a solid rectangle growing along one axis
  shrink   the amount of colour c on the screen goes down. A moving object also leaves a trail of
           slices, but it keeps its colour: what it loses behind it, it gains in front

Whatever passes both for MIN_SLICES actions is a gauge. The part not yet drained is found by walking on
along the axis while the screen shows the fill colour. Its value is then simply how many slices still
show that colour.

Its meaning is learned from what follows: `observe_restart` records what the gauge read when the game
put the player back. A restart with the gauge empty means "empty gauge = life lost". Until the first
such restart a draining gauge is assumed to be a budget (see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .control_net import components, find_motion

Rect = tuple[int, int, int, int]      # y0, x0, y1, x1 (inclusive)


def bounding_rect(points: np.ndarray) -> Rect:
    (y0, x0), (y1, x1) = points.min(0), points.max(0)
    return int(y0), int(x0), int(y1), int(x1)


def area(r: Rect) -> int:
    return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)


def union(a: Rect, b: Rect) -> Rect:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


@dataclass
class Candidate:
    fill: int
    empty: int
    rect: Rect                 # everything drained so far
    slices: int = 1
    last_t: int = 0


@dataclass
class Gauge:
    fill: int
    rect: Rect                 # the whole bar, drained part included
    across: int                # thickness of a slice: pixels of the bar per unit of value
    found_at: int = 0
    restarts: list[int] = field(default_factory=list)   # value read just before each restart

    @property
    def capacity(self) -> int:
        return area(self.rect) // self.across

    def read(self, frame: np.ndarray) -> int:
        y0, x0, y1, x1 = self.rect
        return int((frame[y0:y1 + 1, x0:x1 + 1] == self.fill).sum()) // self.across


class GaugeFinder:
    MIN_SLICES = 3       # slices in a row before a candidate counts as a gauge
    MAX_GAP = 3          # actions a candidate may skip (a blocked move, an action that costs nothing)
    MAX_SLICE = 256      # pixels; anything bigger is a screen change, not a slice

    def __init__(self) -> None:
        self.candidates: list[Candidate] = []
        self.gauge: Gauge | None = None
        self.t = 0
        self.events: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ discovery
    def observe(self, before: np.ndarray, after: np.ndarray) -> list[tuple[str, str]]:
        """Feed one action's frames; returns events (gauge found)."""
        self.events = []
        self.t += 1
        if self.gauge is not None:
            return self.events
        changed = before != after
        if not changed.any() or changed.mean() > 0.3:
            return self.events
        motion = find_motion(before, after)
        residual = changed if motion is None else motion.by_itself     # not explained by the moving object
        for fill, empty in {(int(a), int(b)) for a, b in zip(before[residual], after[residual])}:
            if (after == fill).sum() >= (before == fill).sum():
                continue                                               # the colour moved, it did not shrink
            mask = residual & (before == fill) & (after == empty)
            for points in components(mask):
                rect = bounding_rect(points)
                if len(points) == area(rect) and len(points) <= self.MAX_SLICE:
                    self._add_slice(fill, empty, rect)
        self.candidates = [c for c in self.candidates if self.t - c.last_t <= self.MAX_GAP]
        for c in self.candidates:
            if c.slices >= self.MIN_SLICES:
                self._promote(c, after)
                break
        return self.events

    def _add_slice(self, fill: int, empty: int, rect: Rect) -> None:
        for c in self.candidates:
            if (c.fill, c.empty) != (fill, empty) or c.last_t == self.t:
                continue
            grown = union(c.rect, rect)
            if area(grown) != area(c.rect) + area(rect):               # must stay a solid rectangle
                continue
            along_y = grown[1] == c.rect[1] and grown[3] == c.rect[3]  # same width: it grows downwards/upwards
            along_x = grown[0] == c.rect[0] and grown[2] == c.rect[2]
            if along_y or along_x:
                c.rect, c.slices, c.last_t = grown, c.slices + 1, self.t
                return
        self.candidates.append(Candidate(fill, empty, rect, last_t=self.t))

    def _promote(self, c: Candidate, frame: np.ndarray) -> None:
        y0, x0, y1, x1 = c.rect
        height, width = y1 - y0 + 1, x1 - x0 + 1
        # the bar drains along its long side: a slice of a horizontal bar is as high as the bar
        horizontal = width >= height if width != height else True
        while horizontal:                                              # walk on over the undrained part
            if x1 + 1 < frame.shape[1] and (frame[y0:y1 + 1, x1 + 1] == c.fill).all():
                x1 += 1
            elif x0 > 0 and (frame[y0:y1 + 1, x0 - 1] == c.fill).all():
                x0 -= 1
            else:
                break
        while not horizontal:
            if y1 + 1 < frame.shape[0] and (frame[y1 + 1, x0:x1 + 1] == c.fill).all():
                y1 += 1
            elif y0 > 0 and (frame[y0 - 1, x0:x1 + 1] == c.fill).all():
                y0 -= 1
            else:
                break
        self.gauge = Gauge(c.fill, (y0, x0, y1, x1), height if horizontal else width, self.t)
        self.candidates = []
        self.events.append(("learn", f"gauge found: colour {c.fill}, box {self.gauge.rect}, "
                                     f"capacity {self.gauge.capacity}"))

    # ------------------------------------------------------------------ reading and meaning
    @property
    def capacity(self) -> int | None:
        return None if self.gauge is None else self.gauge.capacity

    def read(self, frame: np.ndarray) -> int | None:
        return None if self.gauge is None else self.gauge.read(frame)

    def observe_restart(self, value_before: int) -> list[tuple[str, str]]:
        """The game put the player back. Remember what the gauge read just before."""
        if self.gauge is None:
            return []
        self.gauge.restarts.append(value_before)
        if value_before == 0:
            return [("learn", "restart with an empty gauge: an empty gauge costs a life")]
        return [("learn", f"restart with {value_before} left in the gauge: something else cost that life")]

    @property
    def fatal_when_empty(self) -> bool | None:
        """None until a restart was seen; then whether restarts come with an empty gauge."""
        if self.gauge is None or not self.gauge.restarts:
            return None
        return 0 in self.gauge.restarts
