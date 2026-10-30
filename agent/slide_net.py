"""SlideNet: a CNN that learns, from one or two surprises, which icons push the player and where to.

In ls20 some cells have a pusher next to them: entering the cell makes the player slide in a fixed
direction until the next wall. The original planner remembered each slide as an exact
(position, action, configuration of every moving block) entry, so a pusher had to be re-tested
every time the moving blocks were somewhere else.

SlideNet learns the pusher from its *look* instead:

  input    15x15 one-hot patch around a cell (the cell in the middle, 5 px of context around it).
           The player is masked out, so the net learns the icon, not "the player was next to it".
  output   P(no push), P(push up), P(push down), P(push left), P(push right)
  end      slide in the predicted direction until the next cell is not free in the visible map.
           A slide that ended somewhere else is remembered as an exception.

Fast learning from surprises: when a move ends somewhere the net did not predict, it immediately takes
FAST_STEPS gradient steps on that move. Every step contains its 8 rotations/reflections (direction
label rotated with it, so one pusher teaches all four orientations) plus earlier examples, so the
net does not forget. Ordinary moves are cheap "no push" examples.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COLORS = 16
CELL = 5
MARGIN = 5
SIZE = CELL + 2 * MARGIN                     # 15 px patch
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   # class 1..4: up, down, left, right
NAMES = ["none", "up", "down", "left", "right"]

Cell = tuple[int, int]


def patch(frame: np.ndarray, cell: Cell, origin: Cell, mask_cell: Cell | None = None) -> np.ndarray:
    """(17, 15, 15) one-hot patch around a grid cell; channel 16 marks pixels outside the frame.

    Pixels of `mask_cell` (the player) are left all-zero: "unknown", neither colour nor outside.
    """
    y0 = origin[0] + CELL * cell[0] - MARGIN
    x0 = origin[1] + CELL * cell[1] - MARGIN
    out = np.zeros((N_COLORS + 1, SIZE, SIZE), np.float32)
    h, w = frame.shape
    ys, xs = np.arange(y0, y0 + SIZE), np.arange(x0, x0 + SIZE)
    inside = (ys[:, None] >= 0) & (ys[:, None] < h) & (xs[None, :] >= 0) & (xs[None, :] < w)
    colours = frame[np.clip(ys, 0, h - 1)][:, np.clip(xs, 0, w - 1)]
    iy, ix = np.nonzero(inside)
    out[colours[iy, ix], iy, ix] = 1.0
    out[N_COLORS][~inside] = 1.0
    if mask_cell is not None:
        my = origin[0] + CELL * mask_cell[0] - y0
        mx = origin[1] + CELL * mask_cell[1] - x0
        out[:, max(0, my):max(0, my + CELL), max(0, mx):max(0, mx + CELL)] = 0.0
    return out


def rotate_direction(d: Cell) -> Cell:
    """Direction after np.rot90 (counter-clockwise) of the image: right -> up, down -> right."""
    return (-d[1], d[0])


def augment(x: np.ndarray, label: int) -> list[tuple[np.ndarray, int]]:
    """The 8 rotations/reflections of a patch, with the push direction transformed the same way."""
    out = []
    for flip in (False, True):
        img = x[:, :, ::-1] if flip else x
        d = None if label == 0 else DIRECTIONS[label - 1]
        if d is not None and flip:
            d = (d[0], -d[1])
        for k in range(4):
            out.append((np.ascontiguousarray(img), 0 if d is None else DIRECTIONS.index(d) + 1))
            img = np.rot90(img, 1, axes=(1, 2))
            d = None if d is None else rotate_direction(d)
    return out


class SlideNet(nn.Module):
    """Small CNN: 17x15x15 patch -> 5 logits (no push, up, down, left, right). Keeps the layout
    (flatten instead of pooling) because which edge the icon is on decides the direction."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_COLORS + 1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16 * SIZE * SIZE, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


@dataclass
class SlideEvent:
    level: int
    cell: Cell                 # the cell the player entered
    observed: int              # class that happened
    predicted: int             # class the net predicted before learning
    p_observed: float          # probability the net gave to what happened
    end: Cell | None           # where the slide ended (pushes only)
    end_predicted: Cell | None
    fast: bool                 # did the surprise trigger fast learning?


class SlideLearner:
    FAST_STEPS = 40          # gradient steps right after a surprise
    STEP_EVERY = 4           # otherwise one step every 4 observed moves
    BATCH_POS = 16           # augmented pushes per batch (pushes are rare; keep them in every batch)
    BATCH_NEG = 16
    LR = 3e-3
    THRESHOLD = 0.5
    CAPACITY = 2_000

    def __init__(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.net = SlideNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.pos: list[tuple[np.ndarray, int]] = []
        self.neg: list[tuple[np.ndarray, int]] = []
        self.exceptions: dict[tuple[int, Cell, int], Cell] = {}   # (level, cell, class) -> observed end
        self.events: list[SlideEvent] = []
        self.n_seen = 0
        self.train_steps = 0
        self.last_table: dict[Cell, tuple[int, float, Cell | None]] = {}

    # ----------------------------------------------------------------- learning
    def observe(self, prev, action_delta: Cell, scene) -> SlideEvent | None:
        """Learn from one move of the planner: prev scene -> scene after moving by action_delta.

        Returns the event for logging, or None when the move says nothing about pushes
        (blocked move, reset, new level).
        """
        if prev is None or scene.level != prev.level or scene.lives < prev.lives:
            return None
        entered = (prev.player[0] + action_delta[0], prev.player[1] + action_delta[1])
        end = scene.player
        if end == prev.player:
            return None                                   # blocked: the entered cell is a wall
        if end == entered:
            label = 0
        else:
            v = (end[0] - entered[0], end[1] - entered[1])
            if (v[0] == 0) == (v[1] == 0):
                return None                               # not a straight slide: not this mechanism
            label = DIRECTIONS.index((int(np.sign(v[0])), int(np.sign(v[1])))) + 1
        x = patch(prev.frame, entered, prev.origin, mask_cell=prev.player)
        probs = self.predict([x])[0]
        predicted = int(probs.argmax()) if self.pos else 0
        end_predicted = None
        if predicted:
            end_predicted = self.slide_end(entered, predicted, prev.free, prev.level)
        if label:
            physics = self.slide_end(entered, label, prev.free, prev.level, use_exceptions=False)
            if physics != end:
                self.exceptions[(prev.level, entered, label)] = end
        surprise = predicted != label or (label and end_predicted != end)
        (self.pos if label else self.neg).append((x, label))
        if len(self.neg) > self.CAPACITY:
            self.neg.pop(self.rng.randrange(len(self.neg)))
        self.n_seen += 1
        fast = bool(surprise and self.pos)
        if fast:
            self.train(self.FAST_STEPS, focus=(x, label))
        elif self.pos and self.n_seen % self.STEP_EVERY == 0:
            self.train(1)
        event = SlideEvent(prev.level, entered, label, predicted, float(probs[label]), end if label else None, end_predicted, fast)
        self.events.append(event)
        return event

    def train(self, steps: int, focus: tuple[np.ndarray, int] | None = None) -> None:
        self.net.train()
        for _ in range(steps):
            batch = list(augment(*focus)) if focus is not None else []
            for _ in range(self.BATCH_POS // 8):
                batch += augment(*self.rng.choice(self.pos))
            for x, y in self.rng.sample(self.neg, min(self.BATCH_NEG, len(self.neg))):
                batch += augment(x, y)[self.rng.randrange(8):][:1]
            xs = torch.from_numpy(np.stack([b[0] for b in batch]))
            ys = torch.tensor([b[1] for b in batch])
            loss = F.cross_entropy(self.net(xs), ys)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.train_steps += 1

    # ---------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, patches: list[np.ndarray]) -> np.ndarray:
        self.net.eval()
        return torch.softmax(self.net(torch.from_numpy(np.stack(patches))), dim=1).numpy()

    def slide_end(self, cell: Cell, label: int, free: set, level: int, use_exceptions: bool = True) -> Cell | None:
        """Where a push in direction `label` from `cell` stops: before the next non-free cell."""
        if use_exceptions and (level, cell, label) in self.exceptions:
            return self.exceptions[(level, cell, label)]
        d = DIRECTIONS[label - 1]
        pos, n = cell, 0
        while (pos[0] + d[0], pos[1] + d[1]) in free and n < 20:
            pos, n = (pos[0] + d[0], pos[1] + d[1]), n + 1
        return pos if n else None

    def table(self, scene) -> dict[Cell, tuple[int, float, Cell | None]]:
        """For every visible free cell: (predicted class, its probability, slide end or None)."""
        if not self.pos:
            self.last_table = {}
            return {}
        cells = sorted(c for c in scene.free if c != scene.player)
        probs = self.predict([patch(scene.frame, c, scene.origin, mask_cell=scene.player) for c in cells])
        out = {}
        for c, p in zip(cells, probs):
            k = int(p.argmax())
            out[c] = (k, float(p[k]), self.slide_end(c, k, scene.free, scene.level) if k else None)
        self.last_table = out
        return out

    def stats(self) -> dict:
        pushes = [e for e in self.events if e.observed]
        first: dict[tuple[int, Cell], SlideEvent] = {}
        for e in pushes:
            first.setdefault((e.level, e.cell), e)
        right = lambda e: e.predicted == e.observed and e.end_predicted == e.end
        return {
            "pushers met": len(first),
            "pushers predicted at first sight": sum(right(e) for e in first.values()),
            "pushes seen": len(pushes),
            "pushes predicted before they happened": sum(e.predicted == e.observed and e.end_predicted == e.end for e in pushes),
            "surprises (fast learning)": sum(e.fast for e in self.events),
            "false alarms": sum(e.observed == 0 and e.predicted != 0 for e in self.events),
            "train steps": self.train_steps,
        }
