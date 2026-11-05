"""A library of small generated games, drawn like ARC-AGI-3 frames, for pretraining WorldNet.

Every game is random: its palette, layout, sizes and whether it has an energy bar. The generator knows
what a real game never tells the agent, and that is what the networks learn from:

  roles      what each pixel is: background, wall, the thing I control, a goal, an item, something
             clickable, a hint, a gauge
  keys       which keys change the game here
  clicks     which pixels change the game when clicked
  distance   how many actions the shortest win still needs

Families. Some are like the development games, some are new, so the networks see more kinds of rules:

  maze       move a block through walls to a door. Optional: a key colour that tiles change and the door
             checks (like ls20), pushers that slide the player to the next wall, refills
  push       push boxes onto targets (new)
  toggle     click tiles that cycle colours. The wanted pattern comes from hint tiles (like ft09), a small
             copy of the target, a mirror image, or "all off" with neighbours toggling too
  move       click an object to select it, then click empty space to move it onto its marker (new: the
             useful click is on empty space)

The held-out games (m0r0, sb26, su15) were not looked at to write this.
"""
from __future__ import annotations

import random
from collections import defaultdict, deque
from itertools import product
from typing import Hashable

import numpy as np

ROLES = ("background", "wall", "player", "goal", "item", "clickable", "hint", "gauge")
R = {name: i for i, name in enumerate(ROLES)}
KEYS = (1, 2, 3, 4, 5, 7)                # key actions; 6 is the click
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
GAMMA = 0.9                              # value of a state = GAMMA ** (actions still needed)
AREA = (4, 60)                           # play area rows and columns, the border is left for gauges


class Canvas:
    def __init__(self, background: int) -> None:
        self.frame = np.full((64, 64), background, np.uint8)
        self.roles = np.zeros((64, 64), np.uint8)

    def rect(self, y: int, x: int, h: int, w: int, colour: int, role: str) -> None:
        y0, x0, y1, x1 = max(0, y), max(0, x), min(64, y + h), min(64, x + w)
        if y1 > y0 and x1 > x0:
            self.frame[y0:y1, x0:x1] = colour
            self.roles[y0:y1, x0:x1] = R[role]

    def outline(self, y: int, x: int, s: int, colour: int, role: str) -> None:
        self.rect(y, x, 1, s, colour, role)
        self.rect(y + s - 1, x, 1, s, colour, role)
        self.rect(y, x, s, 1, colour, role)
        self.rect(y, x + s - 1, s, 1, colour, role)


class Gauge:
    """An energy bar somewhere on the border, at a random fill when drawn."""

    def __init__(self, rng: random.Random, colours: list[int]) -> None:
        self.side = rng.choice(["top", "bottom", "left"])
        self.thick = rng.choice([1, 2])
        self.length = rng.randint(24, 60)
        self.fill, self.empty = colours

    def draw(self, canvas: Canvas, frac: float) -> None:
        full = int(round(self.length * frac))
        for i in range(self.length):
            colour = self.fill if i < full else self.empty
            if self.side == "top":
                canvas.rect(0, 2 + i, self.thick, 1, colour, "gauge")
            elif self.side == "bottom":
                canvas.rect(64 - self.thick, 2 + i, self.thick, 1, colour, "gauge")
            else:
                canvas.rect(2 + i, 0, 1, self.thick, colour, "gauge")


class MiniGame:
    keys: tuple[int, ...] = ()
    clicks = False

    def key(self) -> Hashable: raise NotImplementedError
    def copy(self) -> "MiniGame": raise NotImplementedError
    def step(self, action: int, x: int = -1, y: int = -1) -> None: raise NotImplementedError
    def won(self) -> bool: raise NotImplementedError
    def candidates(self) -> list[tuple[int, int, int]]: raise NotImplementedError
    def draw(self, canvas: Canvas) -> None: raise NotImplementedError
    def click_mask(self) -> np.ndarray: return np.zeros((64, 64), bool)

    def distance(self) -> int | None:
        """Actions the shortest win still needs, if this family can say without a search."""
        return None


def enumerate_states(game: MiniGame, limit: int = 6000):
    """Every state reachable from `game`, and each one's distance to a win (None: cannot win from there)."""
    states, edges, queue = {game.key(): game}, {}, deque([game])
    while queue and len(states) < limit:
        g = queue.popleft()
        k = g.key()
        edges[k] = []
        if g.won():
            continue
        for a, x, y in g.candidates():
            h = g.copy()
            h.step(a, x, y)
            hk = h.key()
            if hk != k:
                edges[k].append(hk)
                if hk not in states:
                    states[hk] = h
                    queue.append(h)
    complete = not queue
    reverse = defaultdict(list)
    for k, outs in edges.items():
        for o in outs:
            reverse[o].append(k)
    dist = {k: 0 for k, g in states.items() if g.won()}
    queue = deque(dist)
    while queue:
        k = queue.popleft()
        for p in reverse[k]:
            if p not in dist:
                dist[p] = dist[k] + 1
                queue.append(p)
    return states, dist, complete


def palette(rng: random.Random, n: int) -> list[int]:
    return rng.sample(range(16), n)


# ------------------------------------------------------------------------------------------- maze
class Maze(MiniGame):
    def __init__(self, rng: random.Random) -> None:
        self.c = rng.choice([3, 4, 5, 6])
        self.n = rng.randint(6, min(12, (AREA[1] - AREA[0]) // self.c))
        self.oy = AREA[0] + rng.randint(0, (AREA[1] - AREA[0]) - self.n * self.c)
        self.ox = AREA[0] + rng.randint(0, (AREA[1] - AREA[0]) - self.n * self.c)
        cols = palette(rng, 12)
        self.bg, self.wallc, self.playerc, self.doorc, self.refillc, self.pushc = cols[:6]
        self.keycols = cols[6:6 + rng.choice([2, 3])]
        self.gauge = Gauge(rng, cols[9:11]) if rng.random() < 0.7 else None
        self.keys = (1, 2, 3, 4) + ((5,) if rng.random() < 0.3 else ())
        n = self.n
        self.wall = np.zeros((n, n), bool)
        self.wall[0, :] = self.wall[-1, :] = self.wall[:, 0] = self.wall[:, -1] = True
        density = rng.uniform(0.05, 0.3)
        for y, x in product(range(1, n - 1), range(1, n - 1)):
            self.wall[y, x] = rng.random() < density
        free = [(y, x) for y, x in product(range(n), range(n)) if not self.wall[y, x]]
        rng.shuffle(free)
        self.pos, self.door = free[0], free[1]
        self.use_key = rng.random() < 0.5
        self.tiles = {}                   # cell -> ("key", colour index) | ("push", action) | ("refill",)
        for cell in free[2:2 + rng.randint(0, 5)]:
            kind = rng.choice(["key", "push", "refill"] if self.use_key else ["push", "refill"])
            if kind == "key":
                self.tiles[cell] = ("key", rng.randrange(len(self.keycols)))
            elif kind == "push":
                self.tiles[cell] = ("push", rng.choice([1, 2, 3, 4]))
            else:
                self.tiles[cell] = ("refill",)
        self.need = rng.randrange(len(self.keycols)) if self.use_key else 0
        self.carry = rng.randrange(len(self.keycols)) if self.use_key else 0

    def key(self):
        return (self.pos, self.carry)

    def copy(self):
        g = object.__new__(Maze)
        g.__dict__.update(self.__dict__)
        return g

    def _free(self, y, x):
        return 0 <= y < self.n and 0 <= x < self.n and not self.wall[y, x]

    def step(self, action, x=-1, y=-1):
        if action not in MOVES:
            return
        dy, dx = MOVES[action]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if not self._free(ny, nx):
            return
        self.pos = (ny, nx)
        for _ in range(self.n):
            tile = self.tiles.get(self.pos)
            if tile is None:
                break
            if tile[0] == "key":
                self.carry = tile[1]
                break
            if tile[0] == "push":
                dy, dx = MOVES[tile[1]]
                moved = False
                while self._free(self.pos[0] + dy, self.pos[1] + dx):
                    self.pos = (self.pos[0] + dy, self.pos[1] + dx)
                    moved = True
                if not moved:
                    break
                continue
            break

    def won(self):
        return self.pos == self.door and (not self.use_key or self.carry == self.need)

    def candidates(self):
        return [(a, -1, -1) for a in (1, 2, 3, 4)]

    def draw(self, canvas):
        c = self.c
        for y, x in product(range(self.n), range(self.n)):
            if self.wall[y, x]:
                canvas.rect(self.oy + y * c, self.ox + x * c, c, c, self.wallc, "wall")
        for (y, x), tile in self.tiles.items():
            py, px = self.oy + y * c, self.ox + x * c
            if tile[0] == "key":
                canvas.rect(py + 1, px + 1, c - 2, c - 2, self.keycols[tile[1]], "item")
            elif tile[0] == "refill":
                canvas.rect(py + c // 2 - 1, px + 1, 2, c - 2, self.refillc, "item")
                canvas.rect(py + 1, px + c // 2 - 1, c - 2, 2, self.refillc, "item")
            else:
                dy, dx = MOVES[tile[1]]
                canvas.rect(py, px, c, c, self.pushc, "item")
                canvas.rect(py + (c - 1) * (dy > 0) if dy else py, px + (c - 1) * (dx > 0) if dx else px,
                            1 if dy else c, 1 if dx else c, self.bg, "item")
        dy, dx = self.door
        canvas.outline(self.oy + dy * c, self.ox + dx * c, c, self.doorc, "goal")
        if self.use_key and c >= 4:
            canvas.rect(self.oy + dy * c + c // 2 - 1, self.ox + dx * c + c // 2 - 1, 2, 2, self.keycols[self.need], "goal")
        py, px = self.pos
        canvas.rect(self.oy + py * c, self.ox + px * c, c, c, self.playerc, "player")
        if self.use_key and c >= 3:
            canvas.rect(self.oy + py * c + c // 2 - (c >= 4), self.ox + px * c + c // 2 - (c >= 4),
                        1 + (c >= 4), 1 + (c >= 4), self.keycols[self.carry], "player")


# ------------------------------------------------------------------------------------------- push
class Push(MiniGame):
    keys = (1, 2, 3, 4)

    def __init__(self, rng: random.Random) -> None:
        self.c = rng.choice([4, 5, 6])
        self.n = rng.randint(6, 8)
        self.oy = AREA[0] + rng.randint(0, (AREA[1] - AREA[0]) - self.n * self.c)
        self.ox = AREA[0] + rng.randint(0, (AREA[1] - AREA[0]) - self.n * self.c)
        cols = palette(rng, 7)
        self.bg, self.wallc, self.playerc, self.boxc, self.targetc = cols[:5]
        self.gauge = Gauge(rng, cols[5:7]) if rng.random() < 0.6 else None
        n = self.n
        self.wall = np.zeros((n, n), bool)
        self.wall[0, :] = self.wall[-1, :] = self.wall[:, 0] = self.wall[:, -1] = True
        for y, x in product(range(1, n - 1), range(1, n - 1)):
            self.wall[y, x] = rng.random() < 0.08
        free = [(y, x) for y, x in product(range(1, n - 1), range(1, n - 1)) if not self.wall[y, x]]
        inner = [(y, x) for y, x in free if 1 < y < n - 2 and 1 < x < n - 2]
        rng.shuffle(free)
        rng.shuffle(inner)
        boxes = 1 if rng.random() < 0.7 or n > 7 else 2
        self.boxes = tuple(sorted(inner[:boxes]))
        self.targets = tuple(sorted([t for t in free if t not in self.boxes][:boxes]))
        rng.shuffle(free)
        self.pos = next(p for p in free if p not in self.boxes and p not in self.targets)

    def key(self):
        return (self.pos, self.boxes)

    def copy(self):
        g = object.__new__(Push)
        g.__dict__.update(self.__dict__)
        return g

    def _free(self, y, x):
        return 0 <= y < self.n and 0 <= x < self.n and not self.wall[y, x]

    def step(self, action, x=-1, y=-1):
        if action not in MOVES:
            return
        dy, dx = MOVES[action]
        ny, nx = self.pos[0] + dy, self.pos[1] + dx
        if not self._free(ny, nx):
            return
        if (ny, nx) in self.boxes:
            by, bx = ny + dy, nx + dx
            if not self._free(by, bx) or (by, bx) in self.boxes:
                return
            self.boxes = tuple(sorted([b for b in self.boxes if b != (ny, nx)] + [(by, bx)]))
        self.pos = (ny, nx)

    def won(self):
        return set(self.boxes) == set(self.targets)

    def candidates(self):
        return [(a, -1, -1) for a in (1, 2, 3, 4)]

    def draw(self, canvas):
        c = self.c
        for y, x in product(range(self.n), range(self.n)):
            if self.wall[y, x]:
                canvas.rect(self.oy + y * c, self.ox + x * c, c, c, self.wallc, "wall")
        for y, x in self.targets:
            canvas.outline(self.oy + y * c, self.ox + x * c, c, self.targetc, "goal")
        for y, x in self.boxes:
            canvas.rect(self.oy + y * c + 1, self.ox + x * c + 1, c - 2, c - 2, self.boxc, "item")
        y, x = self.pos
        canvas.rect(self.oy + y * c, self.ox + x * c, c, c, self.playerc, "player")


# ----------------------------------------------------------------------------------------- toggle
LIGHTS = {}                              # 3x3 "lights out": start pattern -> fewest clicks to all off


def _lights_table():
    if LIGHTS:
        return LIGHTS
    masks = []
    for i in range(9):
        y, x = divmod(i, 3)
        m = 0
        for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            if 0 <= y + dy < 3 and 0 <= x + dx < 3:
                m |= 1 << ((y + dy) * 3 + x + dx)
        masks.append(m)
    for presses in range(512):
        state = 0
        for i in range(9):
            if presses >> i & 1:
                state ^= masks[i]
        n = bin(presses).count("1")
        if state not in LIGHTS or LIGHTS[state] > n:
            LIGHTS[state] = n
    return LIGHTS


class Toggle(MiniGame):
    keys = ()
    clicks = True

    def __init__(self, rng: random.Random) -> None:
        self.rule = rng.choice(["hint", "hint", "copy", "mirror", "lights"])
        self.s = 6 if self.rule == "hint" else rng.choice([4, 5, 6])
        self.gap = rng.choice([1, 2, 3]) if self.rule != "hint" else 2
        k = 2 if self.rule == "lights" else rng.choice([2, 2, 3])
        cols = palette(rng, 9)
        self.bg, self.cycle, self.on, self.off, self.framec = cols[0], cols[1:1 + k], cols[4], cols[5], cols[6]
        self.gauge = Gauge(rng, cols[7:9]) if rng.random() < 0.7 else None
        if self.rule == "lights":
            self.rows = self.cols = 3
        elif self.rule == "mirror":
            self.rows, self.cols = rng.randint(2, 4), 2 * rng.randint(1, 3)
        else:
            self.rows, self.cols = rng.randint(2, 5), rng.randint(2, 5)
        step = self.s + self.gap
        self.oy = AREA[0] + rng.randint(0, max(0, 40 - self.rows * step))
        self.ox = AREA[0] + rng.randint(0, max(0, 50 - self.cols * step))
        self.hints: dict[tuple[int, int], tuple[int, list[bool]]] = {}
        cells = list(product(range(self.rows), range(self.cols)))
        if self.rule == "hint":
            rng.shuffle(cells)
            for cell in cells[:rng.randint(1, max(1, len(cells) // 5))]:
                if all(abs(cell[0] - h[0]) > 1 or abs(cell[1] - h[1]) > 1 for h in self.hints):
                    self.hints[cell] = (0, [])
        self.fixed = {(y, x) for y, x in cells if self.rule == "mirror" and x < self.cols // 2}
        self.clickable = [c for c in sorted(cells) if c not in self.hints and c not in self.fixed]
        target = {c: rng.randrange(k) for c in cells}
        self.target = target
        if self.rule == "hint":
            for h in list(self.hints):
                centre = rng.randrange(k)
                pattern = [target.get((h[0] + dy, h[1] + dx), centre) == centre
                           for dy, dx in product((-1, 0, 1), (-1, 0, 1)) if (dy, dx) != (0, 0)]
                self.hints[h] = (centre, pattern)
        if self.rule == "lights":
            self.target = {c: 0 for c in cells}
        self.colour = {c: rng.randrange(k) for c in self.clickable}
        for c in self.fixed:
            self.colour[c] = target[c]
        self.k = k
        if self.rule == "copy":
            self.panel = (AREA[0] + rng.randint(0, 6), 64 - 4 - self.cols * 2 - 2, rng.choice([1, 2]))

    def key(self):
        return tuple(self.colour[c] for c in self.clickable)

    def copy(self):
        g = object.__new__(Toggle)
        g.__dict__.update(self.__dict__)
        g.colour = dict(self.colour)
        return g

    def _cell_at(self, x, y):
        step = self.s + self.gap
        cy, cx = (y - self.oy) // step, (x - self.ox) // step
        if (y - self.oy) % step < self.s and (x - self.ox) % step < self.s and (cy, cx) in self.colour \
                and (cy, cx) in self.clickable and y >= self.oy and x >= self.ox:
            return cy, cx
        return None

    def step(self, action, x=-1, y=-1):
        if action != 6:
            return
        cell = self._cell_at(x, y)
        if cell is None:
            return
        hit = [cell]
        if self.rule == "lights":
            hit = [(cell[0] + dy, cell[1] + dx) for dy, dx in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))]
        for c in hit:
            if c in self.colour and c in self.clickable:
                self.colour[c] = (self.colour[c] + 1) % self.k

    def _wanted(self, cell) -> list[int]:
        """The colours this cell may take in a win."""
        if self.rule in ("copy", "lights"):
            return [self.target[cell]]
        if self.rule == "mirror":
            y, x = cell
            return [self.colour[(y, self.cols - 1 - x)]]
        ok = set(range(self.k))
        for (hy, hx), (centre, pattern) in self.hints.items():
            offsets = [(dy, dx) for dy, dx in product((-1, 0, 1), (-1, 0, 1)) if (dy, dx) != (0, 0)]
            if (cell[0] - hy, cell[1] - hx) in offsets:
                same = pattern[offsets.index((cell[0] - hy, cell[1] - hx))]
                ok &= {centre} if same else set(range(self.k)) - {centre}
        return sorted(ok)

    def won(self):
        return self.distance() == 0

    def distance(self):
        if self.rule == "lights":
            state = sum(1 << (y * 3 + x) for (y, x) in self.clickable if self.colour[(y, x)] != 0)
            return _lights_table()[state]
        return sum(min((w - self.colour[c]) % self.k for w in self._wanted(c)) for c in self.clickable)

    def candidates(self):
        step = self.s + self.gap
        return [(6, self.ox + x * step + self.s // 2, self.oy + y * step + self.s // 2) for y, x in self.clickable]

    def click_mask(self):
        mask = np.zeros((64, 64), bool)
        step = self.s + self.gap
        for y, x in self.clickable:
            mask[self.oy + y * step:self.oy + y * step + self.s, self.ox + x * step:self.ox + x * step + self.s] = True
        return mask

    def draw(self, canvas):
        step, s = self.s + self.gap, self.s
        for (y, x), value in self.colour.items():
            role = "hint" if (y, x) in self.fixed else "clickable"
            canvas.rect(self.oy + y * step, self.ox + x * step, s, s, self.cycle[value], role)
        if self.rule == "mirror":
            mid = self.ox + (self.cols // 2) * step - self.gap // 2 - 1
            canvas.rect(self.oy - 1, mid, self.rows * step, 1, self.framec, "hint")
        for (y, x), (centre, pattern) in self.hints.items():
            b = s // 3
            py, px = self.oy + y * step, self.ox + x * step
            bits = iter(pattern)
            for j, i in product(range(3), range(3)):
                colour = self.cycle[centre] if (i, j) == (1, 1) else (self.on if next(bits) else self.off)
                canvas.rect(py + j * b, px + i * b, b, b, colour, "hint")
        if self.rule == "copy":
            py, px, b = self.panel
            canvas.outline(py - 1, px - 1, max(self.rows, self.cols) * b + 2, self.framec, "hint")
            for (y, x), value in self.target.items():
                canvas.rect(py + y * b, px + x * b, b, b, self.cycle[value], "hint")


# ------------------------------------------------------------------------------------------- move
class Move(MiniGame):
    keys = ()
    clicks = True

    def __init__(self, rng: random.Random) -> None:
        self.g = rng.choice([6, 7, 8])
        self.n = (AREA[1] - AREA[0]) // self.g
        cols = palette(rng, 8)
        self.bg, self.selc = cols[0], cols[1]
        m = rng.randint(1, 3)
        self.objc = cols[2:2 + m]
        self.gauge = Gauge(rng, cols[6:8]) if rng.random() < 0.5 else None
        self.shapes = []
        for _ in range(m):
            size = rng.randint(3, self.g - 2)
            shape = np.zeros((size, size), bool)
            while shape.sum() < size:
                shape = np.array([[rng.random() < 0.7 for _ in range(size)] for _ in range(size)])
            self.shapes.append(shape)
        cells = list(product(range(self.n), range(self.n)))
        rng.shuffle(cells)
        self.markers = tuple(cells[:m])
        self.at = list(cells[m:2 * m])
        for i in range(m):
            if rng.random() < 0.25:
                self.at[i] = self.markers[i]
        self.at = tuple(self.at)
        self.selected = rng.randrange(-1, m)

    def key(self):
        return (self.at, self.selected)

    def copy(self):
        g = object.__new__(Move)
        g.__dict__.update(self.__dict__)
        return g

    def _object_at(self, x, y):
        for i, (cy, cx) in enumerate(self.at):
            shape = self.shapes[i]
            py, px = AREA[0] + cy * self.g + 1, AREA[0] + cx * self.g + 1
            if py <= y < py + shape.shape[0] and px <= x < px + shape.shape[1] and shape[y - py, x - px]:
                return i
        return None

    def step(self, action, x=-1, y=-1):
        if action != 6:
            return
        hit = self._object_at(x, y)
        if hit is not None:
            self.selected = -1 if hit == self.selected else hit
            return
        cy, cx = (y - AREA[0]) // self.g, (x - AREA[0]) // self.g
        if self.selected >= 0 and 0 <= cy < self.n and 0 <= cx < self.n and (cy, cx) not in self.at:
            at = list(self.at)
            at[self.selected] = (cy, cx)
            self.at, self.selected = tuple(at), -1

    def won(self):
        return all(a == m for a, m in zip(self.at, self.markers))

    def distance(self):
        wrong = [i for i, (a, m) in enumerate(zip(self.at, self.markers)) if a != m]
        return 2 * len(wrong) - (1 if self.selected in wrong else 0)

    def candidates(self):
        return []

    def click_mask(self):
        mask = np.zeros((64, 64), bool)
        for i, (cy, cx) in enumerate(self.at):
            shape = self.shapes[i]
            py, px = AREA[0] + cy * self.g + 1, AREA[0] + cx * self.g + 1
            mask[py:py + shape.shape[0], px:px + shape.shape[1]] |= shape
        if self.selected >= 0:
            for cy, cx in product(range(self.n), range(self.n)):
                if (cy, cx) not in self.at:
                    mask[AREA[0] + cy * self.g:AREA[0] + (cy + 1) * self.g,
                         AREA[0] + cx * self.g:AREA[0] + (cx + 1) * self.g] = True
        return mask

    def draw(self, canvas):
        for i, (cy, cx) in enumerate(self.markers):
            canvas.outline(AREA[0] + cy * self.g, AREA[0] + cx * self.g, self.g, self.objc[i], "goal")
        for i, (cy, cx) in enumerate(self.at):
            shape = self.shapes[i]
            py, px = AREA[0] + cy * self.g + 1, AREA[0] + cx * self.g + 1
            role = "player" if i == self.selected else "clickable"
            for j, k in zip(*np.nonzero(shape)):
                canvas.rect(py + j, px + k, 1, 1, self.objc[i], role)
            if i == self.selected:
                canvas.rect(py - 1, px - 1, 1, shape.shape[1] + 2, self.selc, "player")


FAMILIES = {"maze": Maze, "push": Push, "toggle": Toggle, "move": Move}


# ---------------------------------------------------------------------------------------- samples
def render(game: MiniGame, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    canvas = Canvas(game.bg)
    game.draw(canvas)
    if game.gauge is not None:
        game.gauge.draw(canvas, rng.uniform(0.05, 1.0))
    return canvas.frame, canvas.roles


def labels(game: MiniGame, distance: int | None) -> dict:
    keys = np.zeros(len(KEYS), np.float32)
    available = np.zeros(len(KEYS), np.float32)
    for i, a in enumerate(KEYS):
        if a in game.keys:
            available[i] = 1
            h = game.copy()
            h.step(a)
            keys[i] = float(h.key() != game.key())
    return dict(keys=keys, available=available, clicks=game.click_mask(), can_click=float(game.clicks),
                value=np.float32(0.0 if distance is None else GAMMA ** distance))


def samples(rng: random.Random, per_game: int = 8):
    """A few states of one new random game, with everything the generator knows about them."""
    family = rng.choice(["maze", "maze", "push", "toggle", "toggle", "move"])
    for _ in range(20):
        game = FAMILIES[family](rng)
        if family in ("maze", "push"):
            states, dist, complete = enumerate_states(game)
            usable = [k for k in states if k in dist or complete]
            if game.key() not in dist or len(usable) < 2:
                continue
            picks = [rng.choice(usable) for _ in range(per_game)]
            out = []
            for k in picks:
                g = states[k]
                frame, roles = render(g, rng)
                out.append(dict(frame=frame, roles=roles, family=family, **labels(g, dist.get(k))))
            return out
        out = []
        for _ in range(per_game):
            g = game.copy()
            if family == "toggle":
                for c in g.clickable:
                    g.colour[c] = rng.randrange(g.k)
                if rng.random() < 0.3:              # also show states close to a win
                    for c in g.clickable:
                        g.colour[c] = g._wanted(c)[0] if g._wanted(c) and rng.random() < 0.85 else g.colour[c]
            else:
                g.at = tuple(m if rng.random() < 0.4 else a for a, m in zip(g.at, g.markers))
                if len(set(g.at)) < len(g.at):
                    continue
                g.selected = rng.randrange(-1, len(g.at))
            frame, roles = render(g, rng)
            out.append(dict(frame=frame, roles=roles, family=family, **labels(g, g.distance())))
        if out:
            return out
    return []
