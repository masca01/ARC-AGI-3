"""ProgressNet: a CNN that learns what "close to solving the level" looks like.

Why: exploring blindly works for a first level, but the state space of later levels is too big. Every
solved level leaves something useful behind: the graph of what was tried, ending in the win. From it the
exact distance to the win of every state seen is known, with no game knowledge involved.

  input     a frame without its resource bars
  output    value in 0..1, trained towards GAMMA ** (actions needed to win from here)
  labels    states that can reach the win get that value, states that cannot get 0
  use       on later levels, states that look closer to a win are explored first

It is trained only after a level is solved, so it does nothing during the first one: the first level
pays for what the later ones use. Mirrored and rotated copies add data, since a single level gives few
states and a rule like "this object must reach that one" does not depend on the direction.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import one_hot


class ProgressCNN(nn.Module):
    """Local detail, then global context: being solved can depend on two objects far apart."""

    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(16, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.ReLU(),     # 32x32
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU(),  # 16x16
            nn.Conv2d(2 * width, 2 * width, 3, stride=2, padding=1), nn.ReLU())  # 8x8
        self.head = nn.Sequential(nn.Linear(4 * width, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        return self.head(torch.cat([h.amax(dim=(2, 3)), h.mean(dim=(2, 3))], dim=1)).squeeze(1)


class ProgressLearner:
    GAMMA = 0.95          # value lost per action away from the win
    TRAIN_STEPS = 400     # gradient steps after each solved level
    BATCH = 32
    LR = 1e-3
    CAPACITY = 5_000

    def __init__(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.net = ProgressCNN()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.near: list[tuple[np.ndarray, float]] = []    # states that can reach a win
        self.far: list[tuple[np.ndarray, float]] = []     # states that could not
        self.version = 0                                  # goes up each training: old scores are stale
        self.last_loss: float | None = None

    @property
    def ready(self) -> bool:
        return self.version > 0

    def add_level(self, samples: list[tuple[np.ndarray, int | None]]) -> None:
        """samples: (masked frame, actions to the win, or None if the win could not be reached from it)."""
        for frame, distance in samples:
            if distance is None:
                self.far.append((frame, 0.0))
            else:
                self.near.append((frame, self.GAMMA ** distance))
        self.near, self.far = self.near[-self.CAPACITY:], self.far[-self.CAPACITY:]
        self.train()

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        half = self.BATCH // 2
        picks = [self.rng.choice(self.near) for _ in range(half if self.far else self.BATCH)]
        if self.far:
            picks += [self.rng.choice(self.far) for _ in range(half)]
        frames = []
        for frame, _ in picks:
            frame = np.rot90(frame, self.rng.randrange(4))
            frames.append((np.fliplr(frame) if self.rng.random() < 0.5 else frame).copy())
        return one_hot(np.stack(frames)), torch.tensor([v for _, v in picks], dtype=torch.float32)

    def train(self) -> None:
        if not self.near:
            return
        self.net.train()
        for _ in range(self.TRAIN_STEPS):
            x, y = self._batch()
            loss = F.binary_cross_entropy_with_logits(self.net(x), y)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.last_loss = loss.item()
        self.version += 1

    @torch.no_grad()
    def score(self, frames: list[np.ndarray]) -> np.ndarray:
        self.net.eval()
        return torch.sigmoid(self.net(one_hot(np.stack(frames)))).numpy()
