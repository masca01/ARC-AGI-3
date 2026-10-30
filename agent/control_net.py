"""Neural perception: a CNN that learns what the agent controls, from the agent's own moves.

Nothing about the player's colour, shape or size is hard-coded. The labels come from motion:

  motion label   after a move, find the shift (dy, dx) that best explains the changed pixels
                 ("these pixels reappeared a few pixels further"). The pixels that moved are
                 what I control. Changed pixels that the shift does not explain changed by
                 themselves (step counters, timers, moving blocks).
  CNN            looks at one frame and predicts both masks:
                   control head : pixels of the object I control
                   self head    : pixels that change on their own (the HUD counter)
                 Trained online, inside the game, on the motion labels. Once it has learned what
                 the player looks like it finds the player in any frame - at the start of a new
                 level, after a reset, or after a blocked move - without having to move first.
  locate()       turns the control heatmap into the (grid cell, grid origin) the planner needs.
                 The grid itself is learned from motion: the step is the length of one move and
                 the origin is where moved objects land (pixel position mod step). The CNN only
                 has to say which cell the player is in, so a blob that is a pixel or two off
                 still gives the right cell.
  trusted        the planner gets the CNN only after it passed a self-check: before learning from
                 a labelled move it must already have found the player's cell, several moves in a
                 row. Until then the agent keeps making warm-up moves.
"""
from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 64
N_COLORS = 16
MOVE_IDS = (1, 2, 3, 4)


def one_hot(frames: np.ndarray) -> torch.Tensor:
    """(B, 64, 64) colour indices -> (B, 16, 64, 64) one-hot planes."""
    t = torch.from_numpy(frames.astype(np.int64))
    return F.one_hot(t, N_COLORS).permute(0, 3, 1, 2).float()


def components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean mask, each as an (N, 2) array of (y, x)."""
    seen = np.zeros(mask.shape, dtype=bool)
    out = []
    h, w = mask.shape
    for y0, x0 in np.argwhere(mask):
        if seen[y0, x0]:
            continue
        stack, points = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append(np.array(points))
    return out


# ----------------------------------------------------------------------------- labels
@dataclass
class Motion:
    shift: tuple[int, int]     # (dy, dx) of the moved object
    moved: np.ndarray          # (64, 64) bool, the object's pixels BEFORE the move
    by_itself: np.ndarray      # (64, 64) bool, changed pixels the shift does not explain


def find_motion(before: np.ndarray, after: np.ndarray, max_shift: int = 8, min_pixels: int = 4) -> Motion | None:
    """Best translation explaining the change between two frames, or None if nothing moved.

    A pixel p supports shift d when it changed, p+d changed too, and after[p+d] == before[p]
    (its colour reappeared at p+d). Swapping an object with the floor behind it supports both
    d and -d equally, so each supporting pixel is weighted by how rare its colour is: objects
    are made of rarer colours than the background they move over.
    """
    changed = before != after
    n_changed = int(changed.sum())
    if n_changed < min_pixels or n_changed > 0.3 * before.size:   # nothing happened, or a whole new screen
        return None
    counts = np.bincount(before.ravel(), minlength=N_COLORS).astype(np.float64)
    rarity = -np.log((counts + 1) / counts.sum())
    best, best_score = None, 0.0
    h, w = before.shape
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dy == 0 and dx == 0:
                continue
            # p in the source window, q = p + d in the destination window
            ys, ye, xs, xe = max(0, -dy), min(h, h - dy), max(0, -dx), min(w, w - dx)
            src = before[ys:ye, xs:xe]
            support = (changed[ys:ye, xs:xe] & changed[ys + dy:ye + dy, xs + dx:xe + dx]
                       & (after[ys + dy:ye + dy, xs + dx:xe + dx] == src))
            n = int(support.sum())
            if n < min_pixels:
                continue
            score = float(rarity[src[support]].sum())
            if score > best_score:
                moved = np.zeros_like(changed)
                moved[ys:ye, xs:xe] = support
                best, best_score = ((dy, dx), moved), score
    if best is None:
        return None
    (dy, dx), moved = best
    explained = moved.copy()
    explained[np.roll(np.roll(moved, dy, 0), dx, 1)] = True        # where the object went
    return Motion((dy, dx), moved, changed & ~explained)


# ---------------------------------------------------------------------------- network
class ControlNet(nn.Module):
    """Fully convolutional: one frame in, two per-pixel maps out (control, by-itself).

    Dilated convolutions give a ~31 px receptive field at full resolution, enough to see an
    object together with its surroundings without losing pixel-exact localisation.
    """

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(N_COLORS, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(width, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))                                  # (B, 2, 64, 64) logits


@dataclass
class Sample:
    frame: np.ndarray       # (64, 64) frame before the move
    control: np.ndarray     # (64, 64) bool, pixels of the controlled object
    by_itself: np.ndarray   # (64, 64) bool, pixels that changed on their own
    self_known: bool = True # False when the frame has no observed next move, so no by-itself label


class ControlPerception:
    """Collects motion labels from the agent's moves, trains the CNN, and locates the player."""

    MIN_SAMPLES = 4          # labelled moves before the CNN is trained at all
    WARMUP_TRAIN_STEPS = 15  # gradient steps after a new label until the planner trusts the CNN
    TRAIN_STEPS = 6          # gradient steps after a new label while the CNN is still learning
    TRUST_AFTER = 5          # CNN right on this many labelled moves in a row -> the planner may use it
    CONFIDENT_AFTER = 10     # once the CNN found the player this many labelled moves in a row...
    CONFIDENT_EVERY = 10     # ...it trains only on every 10th label (keeps the agent fast)
    BATCH = 16
    CAPACITY = 2_000
    LR = 2e-3
    POS_WEIGHT = 10.0        # the player is ~1% of the pixels; weigh its pixels up
    THRESHOLD = 0.5

    def __init__(self, seed: int = 0) -> None:
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.net = ControlNet()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.LR)
        self.buffer: list[Sample] = []
        self.shifts: dict[int, Counter] = {}      # action -> Counter of observed shifts
        self.sizes: Counter = Counter()           # (h, w) bounding boxes of the moved object
        self.last_motion_pos: tuple[int, int] | None = None   # top-left of the object after the last move
        self.origins: Counter = Counter()            # grid origins seen in clean motion labels
        self.trusted = False                         # latched once the self-check passed TRUST_AFTER times
        self.checks: deque[bool] = deque(maxlen=self.CONFIDENT_AFTER)   # CNN right before each label?
        self.n_labels = 0
        self.train_steps = 0
        self.last_loss: float | None = None
        self.last_confidence: float | None = None   # mean control probability inside the chosen window

    # ------------------------------------------------------------ learned geometry
    @property
    def step(self) -> int | None:
        """Grid step in pixels: the most common length of one move."""
        lengths = Counter()
        for counter in self.shifts.values():
            for (dy, dx), n in counter.items():
                lengths[max(abs(dy), abs(dx))] += n
        return lengths.most_common(1)[0][0] if lengths else None

    @property
    def size(self) -> tuple[int, int] | None:
        return self.sizes.most_common(1)[0][0] if self.sizes else None

    @property
    def origin(self) -> tuple[int, int] | None:
        """Grid origin: where moved objects usually land, in pixels mod the step."""
        return self.origins.most_common(1)[0][0] if self.origins else None

    @property
    def ready(self) -> bool:
        return len(self.buffer) >= self.MIN_SAMPLES and self.step is not None

    # ---------------------------------------------------------------- learning
    def observe(self, before: np.ndarray, action: int, after: np.ndarray) -> Motion | None:
        """Label one move from motion, store it, and train a little."""
        self.last_motion_pos = None
        if action not in MOVE_IDS:
            return None
        motion = find_motion(before, after)
        if motion is None:
            return None
        usual = self.shifts.get(action)
        if usual and sum(usual.values()) >= 3 and motion.shift != usual.most_common(1)[0][0]:
            return None          # not the usual effect of this action (teleport, another object): no label
        self.shifts.setdefault(action, Counter())[motion.shift] += 1
        ys, xs = np.nonzero(motion.moved)
        box = (int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        self.sizes[box] += 1
        self.last_motion_pos = (int(ys.min() + motion.shift[0]), int(xs.min() + motion.shift[1]))
        if box == self.size:   # partial labels (object over same-coloured pixels) are off by a pixel or two
            step = self.step
            self.origins[(self.last_motion_pos[0] % step, self.last_motion_pos[1] % step)] += 1
            if self.ready:     # self-check against the motion label, before learning from it
                self.checks.append(self.cell(self.top_left(after)) == self.cell(self.last_motion_pos))
                self.trusted |= len(self.checks) >= self.TRUST_AFTER and all(list(self.checks)[-self.TRUST_AFTER:])
        self.n_labels += 1
        self.add(Sample(before, motion.moved, motion.by_itself))
        # the frame after the move shows the object at its new place: a second, free label
        moved_after = np.roll(np.roll(motion.moved, motion.shift[0], 0), motion.shift[1], 1)
        self.add(Sample(after, moved_after, np.zeros_like(moved_after), self_known=False))
        if len(self.buffer) >= self.MIN_SAMPLES:
            if not self.trusted:
                self.train(self.WARMUP_TRAIN_STEPS)
            elif not self.confident:
                self.train(self.TRAIN_STEPS)
            elif self.n_labels % self.CONFIDENT_EVERY == 0:
                self.train(1)
        return motion

    @property
    def confident(self) -> bool:
        return len(self.checks) == self.CONFIDENT_AFTER and all(self.checks)

    def add(self, sample: Sample) -> None:
        if len(self.buffer) >= self.CAPACITY:
            self.buffer.pop(self.rng.randrange(len(self.buffer)))
        self.buffer.append(sample)

    def train(self, steps: int) -> None:
        self.net.train()
        pos_weight = torch.tensor([self.POS_WEIGHT, self.POS_WEIGHT]).view(1, 2, 1, 1)
        for _ in range(steps):
            batch = self.rng.sample(self.buffer, min(self.BATCH, len(self.buffer)))
            x = one_hot(np.stack([s.frame for s in batch]))
            y = torch.from_numpy(np.stack([np.stack([s.control, s.by_itself]) for s in batch]).astype(np.float32))
            known = torch.tensor([[1.0, float(s.self_known)] for s in batch]).view(-1, 2, 1, 1)
            loss = (F.binary_cross_entropy_with_logits(self.net(x), y, pos_weight=pos_weight,
                                                       reduction="none") * known).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.train_steps += 1
            self.last_loss = loss.item()

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(control, by_itself) probability maps for one frame."""
        self.net.eval()
        p = torch.sigmoid(self.net(one_hot(frame[None])))[0].numpy()
        return p[0], p[1]

    @torch.no_grad()
    def layer_maps(self, frame: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """What each conv layer responds to: channel-mean activation, 32x32, scaled to 0..1 (for replays)."""
        self.net.eval()
        x, out, n = one_hot(frame[None]), [], 0
        rf = 1
        for layer in self.net.body:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                rf += (layer.kernel_size[0] - 1) * layer.dilation[0]
            if isinstance(layer, nn.ReLU):
                n += 1
                m = F.avg_pool2d(x.mean(dim=1, keepdim=True), 2)[0, 0].numpy()
                out.append((f"conv{n} · sees {rf}px", m / max(1e-6, float(m.max()))))
        return out

    def top_left(self, frame: np.ndarray) -> tuple[int, int] | None:
        """Pixel top-left corner of the controlled object.

        The object has the learned size h x w, so it is the h x w window of the control map with
        the highest total probability. Stray confident pixels next to it (a HUD symbol of the same
        colours, say) barely move that window, unlike the centre of a thresholded blob.
        """
        if not self.ready:
            return None
        control, _ = self.predict(frame)
        h, w = self.size or (1, 1)
        c = np.pad(control, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        window = c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]     # sum of every h x w window
        y0, x0 = np.unravel_index(int(window.argmax()), window.shape)
        self.last_confidence = float(window[y0, x0]) / (h * w)
        if window[y0, x0] < self.THRESHOLD * h * w / 2:                # no object-like evidence
            return None
        return int(y0), int(x0)

    def cell(self, corner: tuple[int, int] | None) -> tuple[int, int] | None:
        """Nearest grid cell of a pixel top-left corner."""
        if corner is None or self.origin is None:
            return None
        s, (oy, ox) = self.step, self.origin
        return int(round((corner[0] - oy) / s)), int(round((corner[1] - ox) / s))

    def locate(self, frame: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
        """(grid cell, grid origin) of the controlled object, as the planner expects them."""
        if not self.trusted:
            return None
        cell = self.cell(self.top_left(frame))
        return None if cell is None else (cell, self.origin)
