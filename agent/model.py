"""The agent's neural network and how it learns during a game ("the brain").

One CNN looks at the current 64x64 frame and answers three questions at once:

  change head : for each action, and for each pixel I could click, will it
                change the screen AND lead to a screen I have rarely seen?
                (moving back and forth changes the screen but teaches nothing)
  win head    : for each action and click pixel, how strongly did this kind of
                move lead to completing a level before?
  static head : which pixels change no matter what I do (timers, step counters)?
                Those changes do not count as "my action did something".

Everything is learned inside one game from the agent's own experience.
There is no pre-training and no dataset.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 64
N_COLORS = 16
SIMPLE_IDS = (1, 2, 3, 4, 5, 7)                      # network output index -> ACTION id
SIMPLE_INDEX = {a: i for i, a in enumerate(SIMPLE_IDS)}
CLICK_ID = 6                                         # ACTION6 = click at (x, y)


# ----------------------------------------------------------------------------- frames
def to_array(frame_data) -> np.ndarray:
    """Last sub-frame of a FrameData as a (64, 64) uint8 array of colour indices."""
    frames = frame_data.frame if hasattr(frame_data, "frame") else frame_data
    out = np.zeros((GRID, GRID), dtype=np.uint8)
    if not frames:
        return out
    arr = np.asarray(frames[-1], dtype=np.int16)
    h, w = min(arr.shape[0], GRID), min(arr.shape[1], GRID)
    out[:h, :w] = np.clip(arr[:h, :w], 0, N_COLORS - 1)
    return out


def one_hot(frames: np.ndarray) -> torch.Tensor:
    """(B, 64, 64) colour indices -> (B, 16, 64, 64) one-hot planes."""
    t = torch.from_numpy(frames.astype(np.int64))
    return F.one_hot(t, N_COLORS).permute(0, 3, 1, 2).float()


# ---------------------------------------------------------------------------- network
class ActionNet(nn.Module):
    """Frame in, one score per simple action and one score map per click pixel out."""

    def __init__(self) -> None:
        super().__init__()
        n = len(SIMPLE_IDS)
        self.full = nn.Sequential(                                    # 64x64: local detail
            nn.Conv2d(N_COLORS, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(                                    # 8x8: wider context
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU())
        self.simple_head = nn.Linear(256, 2 * n)                      # change + win, per action
        self.map_head = nn.Sequential(                                # change + win + static, per pixel
            nn.Conv2d(32 + 128, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.full(x)
        g = self.down(f)
        pooled = torch.cat([g.amax(dim=(2, 3)), g.mean(dim=(2, 3))], dim=1)
        s = self.simple_head(pooled).view(-1, 2, len(SIMPLE_IDS))
        up = F.interpolate(g, size=(GRID, GRID), mode="nearest")
        m = self.map_head(torch.cat([f, up], dim=1))
        return {"change_simple": s[:, 0], "win_simple": s[:, 1],
                "change_click": m[:, 0], "win_click": m[:, 1], "static": m[:, 2]}


# ----------------------------------------------------------------------------- memory
@dataclass
class Transition:
    key: bytes
    frame: np.ndarray        # (64, 64) frame before the action
    action: int              # ACTION id
    x: int                   # click position, -1 for simple actions
    y: int
    changed: np.ndarray      # (64, 64) bool, pixels that differ in the next frame
    next_key: bytes          # identity of the resulting screen (counter pixels ignored)
    win: float = 0.0         # credit given when the attempt containing it completed a level


class Brain:
    MIN_SAMPLES = 16          # start training after this many unique transitions
    TRAIN_EVERY = 4           # one training round every 4 moves (also when moves repeat)
    TRAIN_BATCHES = 2         # mini-batches per training round
    WIN_TRAIN_BATCHES = 10    # extra mini-batches right after a level is completed
    BATCH = 32
    CAPACITY = 20_000
    LR = 1e-3
    GAMMA = 0.9               # win credit shrinks by this factor per step before the win
    UNEXPECTED_PIXELS = 2.0   # a change counts if at least this many pixels changed unexpectedly

    def __init__(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.net = ActionNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.buffer: list[Transition] = []
        self.index: dict[bytes, Transition] = {}   # same frame + same action is stored once
        self.attempt: list[Transition] = []        # moves since the last RESET or level start
        self.visits: dict[bytes, int] = {}         # how often each screen was reached
        self.ignore = np.zeros((GRID, GRID), bool) # pixels the static head ever flagged (counters)
        self.n_moves = 0
        self.n_added = 0
        self.train_steps = 0
        self.last_loss: dict[str, float] = {}

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= self.MIN_SAMPLES

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Probabilities for every head on one frame."""
        self.net.eval()
        out = self.net(one_hot(frame[None]))
        return {k: torch.sigmoid(v[0]).numpy() for k, v in out.items()}

    def add(self, frame: np.ndarray, action: int, x: int, y: int, next_frame: np.ndarray,
            p_static: np.ndarray | None = None) -> float:
        """Store one move. Returns its change-head target: meaningful change x novelty (0..1).

        p_static: the static-head prediction for `frame`, if the agent already computed it.
        """
        if p_static is None:
            p_static = self.predict(frame)["static"] if self.ready else np.full((GRID, GRID), 0.5)
        changed = frame != next_frame
        if self.ready:
            self.ignore |= p_static > 0.5
        screen = np.where(self.ignore, 0, next_frame).astype(np.uint8)       # hide counter pixels
        next_key = hashlib.blake2b(screen.tobytes(), digest_size=16).digest()
        self.visits[next_key] = self.visits.get(next_key, 0) + 1
        key = hashlib.blake2b(frame.tobytes() + f"{action},{x},{y}".encode(), digest_size=16).digest()
        t = self.index.get(key)
        if t is None:
            t = Transition(key, frame, action, x, y, changed, next_key)
            if len(self.buffer) >= self.CAPACITY:
                old = self.buffer.pop(random.randrange(len(self.buffer)))
                self.index.pop(old.key, None)
            self.buffer.append(t)
            self.index[key] = t
            self.n_added += 1
        self.attempt.append(t)
        self.n_moves += 1
        if self.ready and self.n_moves % self.TRAIN_EVERY == 0:
            self.train(self.TRAIN_BATCHES)
        meaningful = float((changed * (1.0 - p_static)).sum()) >= self.UNEXPECTED_PIXELS
        return float(meaningful) * self.visits[next_key] ** -0.5

    def level_completed(self) -> None:
        """Give credit to the moves of the winning attempt: the last move gets 1, earlier ones less."""
        for steps_before_win, t in enumerate(reversed(self.attempt)):
            t.win = max(t.win, self.GAMMA ** steps_before_win)
        self.attempt.clear()
        if self.ready:
            self.train(self.WIN_TRAIN_BATCHES)

    def attempt_failed(self) -> None:
        self.attempt.clear()

    def train(self, batches: int = 1) -> None:
        self.net.train()
        for _ in range(batches):
            batch = random.sample(self.buffer, min(self.BATCH, len(self.buffer)))
            out = self.net(one_hot(np.stack([t.frame for t in batch])))
            changed = torch.from_numpy(np.stack([t.changed for t in batch]).astype(np.float32))

            # static head: which pixels change regardless of the action
            static_loss = F.binary_cross_entropy_with_logits(out["static"], changed)
            p_static = torch.sigmoid(out["static"]).detach()
            meaningful = ((changed * (1 - p_static)).sum(dim=(1, 2)) >= self.UNEXPECTED_PIXELS).float()

            # pick the output of the action that was actually taken
            i = torch.arange(len(batch))
            is_click = torch.tensor([t.action == CLICK_ID for t in batch])
            a = torch.tensor([SIMPLE_INDEX.get(t.action, 0) for t in batch])
            xs = torch.tensor([max(t.x, 0) for t in batch])
            ys = torch.tensor([max(t.y, 0) for t in batch])

            def taken(simple: torch.Tensor, click: torch.Tensor) -> torch.Tensor:
                return torch.where(is_click, click[i, ys, xs], simple[i, a])

            # target = changed meaningfully x novelty of the screen it led to (1, 0.71, 0.58, 0.5, ...)
            novelty = torch.tensor([self.visits.get(t.next_key, 1) ** -0.5 for t in batch])
            change_loss = F.binary_cross_entropy_with_logits(
                taken(out["change_simple"], out["change_click"]), meaningful * novelty)
            win_target = torch.tensor([t.win for t in batch])
            win_weight = 1.0 + 9.0 * (win_target > 0).float()   # wins are rare, weigh them more
            win_loss = (F.binary_cross_entropy_with_logits(
                taken(out["win_simple"], out["win_click"]), win_target, reduction="none") * win_weight).mean()

            loss = change_loss + win_loss + static_loss
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.train_steps += 1
            self.last_loss = {"change": change_loss.item(), "win": win_loss.item(), "static": static_loss.item()}
