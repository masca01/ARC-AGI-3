"""CellNet: learns from a solved level which objects are in the wrong state.

A solved level shows many objects of one shape that end up the way the level wants them (cells). Recolouring
some of them at random gives examples of wrong cells, so one level is enough to train on. Each cell is seen
from its own point of view: a crop centred on it, where a pixel counts as "my colour" or "another cell colour"
rather than by its name. That is what lets the rule carry to a level with new colours and a new layout.

Nothing here knows what cells or levels mean in a particular game.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CROP = 32                # pixels seen around a cell
BACKGROUND_AREA = 0.1    # a colour covering this share of the frame is background, unless it is on a cell
HIDDEN = -1              # pixels the agent ignores
CHANNELS = 19            # my colour, another cell colour, background, and the 16 fixed colours


class CellCNN(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(CHANNELS, width, 3, padding=1), nn.ReLU(), nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        side = CROP // 8
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(2 * width * side * side, 128), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(views)).squeeze(1)


@dataclass
class SolvedLevel:
    final: np.ndarray                    # the level as it looked solved, ignored pixels are HIDDEN
    cells: list[tuple[np.ndarray, tuple[int, int, int, int]]]   # (pixels, box y0 x0 y1 x1)
    colours: list[int]                   # the colours a cell can take


def view(frame: np.ndarray, box: tuple[int, int, int, int], cell_colours: set[int], fixed: set[int]) -> np.ndarray:
    """What one cell sees: CROP x CROP pixels centred on it, described relative to it."""
    y0, x0, y1, x1 = box
    top, left = (y0 + y1) // 2 - CROP // 2 + 1, (x0 + x1) // 2 - CROP // 2 + 1
    pad = CROP
    crop = np.pad(frame, pad, constant_values=HIDDEN)[top + pad:top + pad + CROP, left + pad:left + pad + CROP]
    me = frame[(y0 + y1) // 2, (x0 + x1) // 2]
    areas = {int(c): int((frame == c).sum()) for c in np.unique(frame)}
    channels = np.zeros((CHANNELS, CROP, CROP), np.float32)
    for colour in np.unique(crop):
        pixels = crop == colour
        colour = int(colour)
        if colour == me:
            channels[0][pixels] = 1
        elif colour in fixed:
            channels[3 + colour][pixels] = 1
        elif colour in cell_colours:
            channels[1][pixels] = 1
        elif colour == HIDDEN or areas[colour] > BACKGROUND_AREA * frame.size:
            channels[2][pixels] = 1
        else:
            channels[1][pixels] = 1
    return channels


class CellLearner:
    TRAIN_STEPS = 1000
    BATCH = 32
    LR = 1e-3
    GOOD_LOSS = 0.005    # stop training once the running loss is this low
    NETS = 3             # one solved level is little evidence: average a few nets that started differently

    def __init__(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.nets = [CellCNN() for _ in range(self.NETS)]
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=self.LR) for net in self.nets]
        self.levels: list[SolvedLevel] = []
        self.fixed: set[int] = set()         # colours that mean the same in every level
        self.last_loss: float | None = None

    @property
    def ready(self) -> bool:
        return bool(self.levels)

    def add_level(self, final: np.ndarray, cells: list, colours: list[int]) -> None:
        self.levels.append(SolvedLevel(final, cells, colours))
        for colour in np.unique(final):
            colour = int(colour)
            if colour != HIDDEN and colour not in colours and (final == colour).mean() <= BACKGROUND_AREA:
                self.fixed.add(colour)

    def _example(self) -> tuple[np.ndarray, float]:
        level = self.rng.choice(self.levels)
        frame = level.final.copy()
        wrong = []
        for pixels, _ in level.cells:
            flip = self.rng.random() < 0.5
            wrong.append(flip)
            if flip:
                was = int(frame[pixels][0])
                frame[pixels] = self.rng.choice([c for c in level.colours if c != was])
        chosen = self.rng.randrange(len(level.cells))
        seen = view(frame, level.cells[chosen][1], set(level.colours), self.fixed)
        seen = np.rot90(seen, self.rng.randrange(4), (1, 2))
        if self.rng.random() < 0.5:
            seen = seen[:, :, ::-1]
        return seen.copy(), float(wrong[chosen])

    def train(self) -> None:
        for net, optimizer in zip(self.nets, self.optimizers):
            running = None
            for _ in range(self.TRAIN_STEPS):
                batch = [self._example() for _ in range(self.BATCH)]
                views = torch.from_numpy(np.stack([b[0] for b in batch]))
                labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
                loss = F.binary_cross_entropy_with_logits(net(views), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running = loss.item() if running is None else 0.9 * running + 0.1 * loss.item()
                if running < self.GOOD_LOSS:
                    break
            self.last_loss = running

    def wrong(self, frame: np.ndarray, boxes: list[tuple[int, int, int, int]], colours: set[int]) -> np.ndarray:
        """P(this cell is not the way the level wants it), per cell."""
        if not boxes:
            return np.zeros(0)
        views = torch.from_numpy(np.stack([view(frame, box, colours, self.fixed) for box in boxes]))
        for net in self.nets:
            net.eval()
        with torch.no_grad():
            scores = torch.stack([torch.sigmoid(net(views)) for net in self.nets]).mean(0).numpy()
        for net in self.nets:
            net.train()
        return scores
