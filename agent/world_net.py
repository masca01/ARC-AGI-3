"""WorldNet: one CNN pretrained on a library of generated games (pretrain/library.py).

It is what the agent "already knows" before it sees a real game, in the order a person would think:

  keys     which of the keys will change something here          -> which controls to try first
  clicks   where a click will change something, per pixel         -> where to click, also on empty space
  roles    what each pixel is: background, wall, what I control, a goal, an item, something clickable,
           a hint, a gauge                                        -> what may matter in this game
  value    how close this screen is to a win (GAMMA ** actions)   -> which screen to explore from

The trunk keeps the full 64 x 64 resolution and widens its view with dilated convolutions, so a pixel's
prediction can depend on things far from it (a hint 10 pixels away, the door at the other end).
In a real game the agent fine-tunes the value head on the levels it solves.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COLOURS = 16
N_ROLES = 8
KEYS = (1, 2, 3, 4, 5, 7)
WEIGHTS = Path(__file__).resolve().parent / "weights" / "world_net.pt"
ROLE_NAMES = ("background", "wall", "player", "goal", "item", "clickable", "hint", "gauge")


class Block(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.norm = nn.GroupNorm(4, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(F.relu(self.norm(self.conv1(x))))


class WorldNet(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        ramp = torch.linspace(-1, 1, 64)
        self.register_buffer("where", torch.stack([ramp[:, None].expand(64, 64), ramp[None, :].expand(64, 64)]))
        self.stem = nn.Conv2d(N_COLOURS + 2, width, 3, padding=1)
        self.trunk = nn.Sequential(*[Block(width, d) for d in (1, 2, 4, 8, 16, 1)])
        self.roles = nn.Conv2d(width, N_ROLES, 1)
        self.clicks = nn.Conv2d(width, 1, 1)
        self.pooled = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU())
        self.keys = nn.Linear(64, len(KEYS))
        self.value = nn.Linear(64, 1)

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.one_hot(frames.long(), N_COLOURS).permute(0, 3, 1, 2).float()
        x = torch.cat([x, self.where.expand(len(frames), -1, -1, -1)], 1)
        h = self.trunk(F.relu(self.stem(x)))
        pooled = self.pooled(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        return {"roles": self.roles(h), "clicks": self.clicks(h).squeeze(1),
                "keys": self.keys(pooled), "value": self.value(pooled).squeeze(1)}


class World:
    """The pretrained net as the agent uses it, plus fine-tuning of the value head inside a game."""
    FINE_TUNE_STEPS = 150
    LR = 3e-4

    def __init__(self, path: Path = WEIGHTS) -> None:
        self.net = WorldNet()
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()
        self.examples: list[tuple[np.ndarray, float]] = []
        self.version = 0                         # goes up when the value head changed
        self.optimizer = torch.optim.Adam(list(self.net.pooled.parameters()) + list(self.net.value.parameters()),
                                          lr=self.LR)

    @torch.no_grad()
    def predict(self, frames: np.ndarray) -> dict[str, np.ndarray]:
        out = self.net(torch.from_numpy(np.asarray(frames, np.int64)))
        return {"roles": out["roles"].softmax(1).numpy(), "clicks": torch.sigmoid(out["clicks"]).numpy(),
                "keys": torch.sigmoid(out["keys"]).numpy(), "value": torch.sigmoid(out["value"]).numpy()}

    @torch.no_grad()
    def _features(self, frames: np.ndarray) -> torch.Tensor:
        """The trunk's pooled view of each frame: the input of the value head."""
        net, out = self.net, []
        for i in range(0, len(frames), 64):
            x = torch.from_numpy(np.asarray(frames[i:i + 64], np.int64))
            x = F.one_hot(x, N_COLOURS).permute(0, 3, 1, 2).float()
            x = torch.cat([x, net.where.expand(len(x), -1, -1, -1)], 1)
            h = net.trunk(F.relu(net.stem(x)))
            out.append(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))
        return torch.cat(out)

    def add_level(self, frames_and_values: list[tuple[np.ndarray, float]]) -> None:
        """States of a solved level with how close each was to the win: adapt the value head to this game.
        Only the small head trains, on features computed once, so this takes a second, not minutes."""
        if not frames_and_values:
            return
        self.examples += frames_and_values
        features = self._features(np.stack([f for f, _ in self.examples]))
        targets = torch.tensor([v for _, v in self.examples], dtype=torch.float32)
        rng = np.random.default_rng(len(self.examples))
        for _ in range(self.FINE_TUNE_STEPS):
            i = torch.from_numpy(rng.integers(0, len(features), 32))
            value = self.net.value(self.net.pooled(features[i])).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(value, targets[i])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.version += 1
