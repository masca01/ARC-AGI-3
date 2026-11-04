"""GeneralAgent: plays any ARC-AGI-3 game with no knowledge of a particular game.

It assumes only what is true of every game: frames go in, actions come out, and the same action in the
same situation does the same thing. Nothing here knows about keys, doors, players or colours.

  state      a frame, without the pixels that do not describe the situation: resource bars (gauge.py) and
             whatever differs between two starts of the same level (a life icon)
  actions    the keys the game offers, plus one click per object on screen (a connected patch of colour)
  graph      every state seen in this level, and where each action led. The agent tries an untried action
             here, or walks through the graph to the nearest state that has one
  budget     walks are planned with the gauge: a route must be affordable, and an action that refilled
             the gauge is a known way to go further
  transfer   objects are grouped by appearance. What a kind of object did (react or not) is kept across
             levels, so later levels skip what never reacted. Each solved level also trains ProgressNet
             (progress_net.py), and later levels explore first where it sees a state close to a win
  cells      when a solved level was about changing the colour of same-shaped objects, it trains CellNet
             (cell_net.py), and later levels click the objects it sees as wrong before exploring

Contract (same as the official framework and Kaggle):
  class GeneralAgent(Agent) with is_done(frames, latest_frame) and choose_action(frames, latest_frame)
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Hashable

import numpy as np
from arcengine import FrameData, GameAction, GameState
from scipy import ndimage

try:  # inside the official ARC-AGI-3-Agents framework (and on Kaggle)
    from agents.agent import Agent  # type: ignore
except ImportError:  # local development in this repo
    from .base import Agent

from .cell_net import HIDDEN, CellLearner
from .gauge import GaugeFinder
from .model import CLICK_ID, to_array
from .progress_net import ProgressLearner

ActionKey = Hashable                     # an ACTION id, or (x, y) for a click
FOUR_NEIGHBOURS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


@dataclass
class Target:
    """One thing the agent can do in a state."""
    action: ActionKey
    kind: Hashable                       # what kind of action or object this is
    box: tuple[int, int, int, int] | None = None     # y0 x0 y1 x1, for an object
    pixels: np.ndarray | None = None     # its pixels in the frame, for an object


def objects(frame: np.ndarray, ignore: np.ndarray, max_area: float = 0.25) -> list[Target]:
    """One click per connected patch of colour. The big patches are background."""
    targets = []
    for colour in np.unique(frame[~ignore]):
        labels, n = ndimage.label((frame == colour) & ~ignore, structure=FOUR_NEIGHBOURS)
        for i, box in enumerate(ndimage.find_objects(labels), start=1):
            mask = labels[box] == i
            if mask.sum() > max_area * frame.size:
                continue
            ys, xs = np.nonzero(mask)
            middle = len(ys) // 2                  # a pixel of the object itself, not its bounding-box centre
            y, x = int(ys[middle] + box[0].start), int(xs[middle] + box[1].start)
            shape = hashlib.blake2b(np.packbits(mask).tobytes() + bytes(mask.shape), digest_size=6).digest()
            whole = np.zeros(frame.shape, bool)
            whole[box][mask] = True
            corners = (box[0].start, box[1].start, box[0].stop - 1, box[1].stop - 1)
            targets.append(Target((x, y), (int(colour), shape), corners, whole))
    return targets


class KindStats:
    """How often each kind of action or object reacted, kept across levels."""
    GIVE_UP_AFTER = 3     # tries without any reaction before a kind is left for last

    def __init__(self) -> None:
        self.tries: dict[Hashable, int] = {}
        self.hits: dict[Hashable, int] = {}

    def update(self, kind: Hashable, reacted: bool) -> None:
        self.tries[kind] = self.tries.get(kind, 0) + 1
        self.hits[kind] = self.hits.get(kind, 0) + int(reacted)

    def priority(self, kind: Hashable) -> float:
        """Higher first. Unknown kinds rank between what reacts and what never did."""
        tries, hits = self.tries.get(kind, 0), self.hits.get(kind, 0)
        if tries == 0:
            return 0.5
        if hits == 0 and tries >= self.GIVE_UP_AFTER:
            return -1.0
        return (hits + 0.5) / (tries + 1)


@dataclass
class Step:
    """One action as it happened. Kept raw so the graph can be rebuilt when the state definition changes."""
    before: np.ndarray
    action: ActionKey
    after: np.ndarray
    spent: int                           # gauge before minus gauge after


@dataclass
class Node:
    targets: list[Target]
    frame: np.ndarray                    # as the networks see it: without the ignored pixels
    edges: dict[ActionKey, tuple[bytes, int]] = field(default_factory=dict)   # action -> (next state, spent)
    fatal: set[ActionKey] = field(default_factory=set)                        # actions that ended the attempt
    value: float = 0.0                   # ProgressNet's opinion
    value_version: int = -1


class LevelGraph:
    def __init__(self) -> None:
        self.nodes: dict[bytes, Node] = {}
        self.start: bytes | None = None

    def untried(self, key: bytes) -> list[Target]:
        node = self.nodes[key]
        return [t for t in node.targets if t.action not in node.edges and t.action not in node.fatal]

    def route(self, source: bytes, wanted, energy: int | None) -> list[ActionKey] | None:
        """Shortest known route to a state where wanted(state). With a gauge, it must be affordable:
        a state is searched again only if it can be reached with more energy left."""
        best = {source: energy if energy is not None else 0}
        queue = deque([(source, best[source], [])])
        while queue:
            key, left, path = queue.popleft()
            if path and wanted(key):
                return path
            for action, (target, spent) in self.nodes[key].edges.items():
                if target == key or target not in self.nodes:
                    continue
                remaining = left
                if energy is not None:
                    remaining = left - spent
                    if remaining <= 0:
                        continue
                if target not in best or remaining > best[target]:
                    best[target] = remaining
                    queue.append((target, remaining, path + [action]))
        return None

    def distances(self, source: bytes) -> dict[bytes, list[ActionKey]]:
        """Shortest known route from `source` to every reachable state."""
        routes: dict[bytes, list[ActionKey]] = {source: []}
        queue = deque([source])
        while queue:
            key = queue.popleft()
            for action, (target, _) in self.nodes[key].edges.items():
                if target not in routes and target in self.nodes:
                    routes[target] = routes[key] + [action]
                    queue.append(target)
        return routes

    def distance_to(self, goal: bytes) -> dict[bytes, int]:
        """Actions needed to reach `goal` from every state that can."""
        incoming: dict[bytes, list[bytes]] = {}
        for key, node in self.nodes.items():
            for target, _ in node.edges.values():
                incoming.setdefault(target, []).append(key)
        distance, queue = {goal: 0}, deque([goal])
        while queue:
            key = queue.popleft()
            for source in incoming.get(key, []):
                if source not in distance:
                    distance[source] = distance[key] + 1
                    queue.append(source)
        return distance


class GeneralAgent(Agent):
    MAX_ACTIONS = 2000
    SEED = 0
    PROGRESS = False         # let ProgressNet order the exploration once a level was solved (no gain so far)
    CELLS = True             # let CellNet choose clicks once a level about object colours was solved
    MIN_CELLS = 4            # fewer cells than this in a solved level are not worth training on
    CELL_CLICKS = 4          # clicks on one cell before CellNet leaves it alone in this attempt
    DETOUR_COST = 0.02       # value a state must offer per extra action of walking to it
    LOW_GAUGE = 0.1          # a death below this share of the gauge is blamed on the gauge, not the action
    SAME_START = 0.01        # two frames this close (share of pixels) are the same start of a level

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gauge = GaugeFinder()
        self.rng = random.Random(self.SEED)
        self.kinds = KindStats()                 # by appearance, kept across levels
        self.places = KindStats()                # by place on screen, this level only
        self.why_counts: Counter = Counter()
        self.progress = ProgressLearner() if self.PROGRESS else None
        self.cell_learner = CellLearner(self.SEED) if self.CELLS else None
        self.cell_shapes: set[Hashable] = set()  # shapes CellNet was trained on
        self.cell_clicks: Counter = Counter()    # this attempt: clicks CellNet asked for, by place
        self.dead_cells: set[ActionKey] = set()  # this level: places that did not react to CellNet's click
        self.doubted: set[ActionKey] = set()     # this level: cells tried against CellNet's opinion
        self.ignore = np.zeros((64, 64), bool)   # pixels that do not describe the situation, this level
        self.graph = LevelGraph()
        self.history: list[Step] = []            # this level, for rebuilding the graph
        self.level_start: np.ndarray | None = None
        self.prev: tuple[np.ndarray, ActionKey, bytes, int | None] | None = None
        self.route: list[ActionKey] = []
        self._expected: bytes | None = None
        self.restarting = False                  # the next frame is a start of this level
        self.midway = False                      # took over inside a level: no start seen yet
        self.level = 0
        self.level_actions = [0]
        self.resets = 0
        self.last_choice: dict[str, Any] = {}
        self.trace: list[dict] = []

    # ----------------------------------------------------------------- contract
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        frame = to_array(latest_frame)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._close_step(frame, died=True)
            self.resets += latest_frame.state is GameState.GAME_OVER
            self.route = []
            self.restarting = True
            self.cell_clicks.clear()
            self.last_choice = {"action": "RESET", "why": "start of game or game over"}
            return GameAction.RESET

        level = latest_frame.levels_completed
        if level > self.level:
            self._close_step(frame, solved=True)
            self._new_level(level)
        else:
            self._close_step(frame)
        self.midway = self.midway and not self.restarting
        if self.level_start is None:
            self.level_start = None if self.midway else frame
        elif self.restarting:
            self._compare_starts(frame)
        self.restarting = False

        key = self.state(frame)
        node = self._node(key, frame, latest_frame)
        if self.graph.start is None:
            self.graph.start = key
        action, why = self._decide(key, node, frame)
        self.prev = (frame, action, key, self.gauge.read(frame))
        self.level_actions[self.level] += 1
        self.last_choice = {"action": str(action), "why": why}
        self.why_counts[why] += 1
        return self._game_action(action)

    def take_over(self, level: int, midway: bool) -> None:
        """Start playing a game someone else began. A frame from inside a level must not count as its start."""
        self._new_level(level)
        self.midway = midway

    # ----------------------------------------------------------------- state
    def state(self, frame: np.ndarray) -> bytes:
        return hashlib.blake2b(np.where(self._mask(), 0, frame + 1).astype(np.int8).tobytes(), digest_size=16).digest()

    def _mask(self) -> np.ndarray:
        mask = self.ignore.copy()
        if self.gauge.gauge is not None:
            y0, x0, y1, x1 = self.gauge.gauge.rect
            mask[y0:y1 + 1, x0:x1 + 1] = True
        return mask

    def _seen_frame(self, frame: np.ndarray) -> np.ndarray:
        return np.where(self._mask(), 0, frame).astype(np.uint8)

    def _compare_starts(self, frame: np.ndarray) -> None:
        """Two starts of one level differ only in what does not describe it (a life icon, a counter)."""
        different = (frame != self.level_start) & ~self._mask()
        if different.any() and different.mean() < self.SAME_START:
            self.ignore |= different
            self._rebuild()

    def _node(self, key: bytes, frame: np.ndarray, latest_frame: FrameData) -> Node:
        node = self.graph.nodes.get(key)
        if node is None:
            available = latest_frame.available_actions or []
            targets = [Target(a, ("key", a)) for a in available if a != CLICK_ID]
            if CLICK_ID in available:
                targets += objects(frame, self._mask())
            self.rng.shuffle(targets)          # ties are broken by order, so runs differ only by seed
            node = self.graph.nodes[key] = Node(targets, self._seen_frame(frame))
        return node

    def _rebuild(self) -> None:
        """The definition of a state changed: replay this level's steps into a new graph."""
        old, self.graph = self.graph, LevelGraph()
        self.route = []
        for step in self.history:
            key, target = self.state(step.before), self.state(step.after)
            if key not in self.graph.nodes:
                self.graph.nodes[key] = Node(self._targets_like(old, step.before), self._seen_frame(step.before))
            if target not in self.graph.nodes:
                self.graph.nodes[target] = Node(self._targets_like(old, step.after), self._seen_frame(step.after))
            self.graph.nodes[key].edges[step.action] = (target, step.spent)
        if self.level_start is not None:
            self.graph.start = self.state(self.level_start)

    def _targets_like(self, old: LevelGraph, frame: np.ndarray) -> list[Target]:
        first = next(iter(old.nodes.values()), None)
        simple = [t for t in first.targets if not isinstance(t.action, tuple)] if first else []
        clicks = any(isinstance(t.action, tuple) for n in old.nodes.values() for t in n.targets)
        return simple + (objects(frame, self._mask()) if clicks else [])

    def _new_level(self, level: int) -> None:
        while len(self.level_actions) <= level:
            self.level_actions.append(0)
        self.level = level
        self.graph, self.history, self.route = LevelGraph(), [], []
        self.places = KindStats()
        self.cell_clicks.clear()
        self.dead_cells.clear()
        self.doubted.clear()
        self.ignore = np.zeros((64, 64), bool)
        self.level_start = None
        self.prev = None

    # ----------------------------------------------------------------- learning
    def _close_step(self, frame: np.ndarray, died: bool = False, solved: bool = False) -> None:
        if self.prev is None:
            return
        before, action, key, value = self.prev
        self.prev = None
        self.gauge.observe(before, frame)
        after_value = self.gauge.read(frame)
        node = self.graph.nodes[key]
        kind = next((t.kind for t in node.targets if t.action == action), None)
        target = self.state(frame)
        if kind is not None:
            self.kinds.update(kind, target != key)
            self.places.update(action, target != key)
        if target == key and action in self.cell_clicks:
            self.dead_cells.add(action)
        if died and self.gauge.gauge is not None and value and value > self.LOW_GAUGE * self.gauge.capacity:
            node.fatal.add(action)             # not out of energy: this action itself is deadly here
        if solved:
            self._learn_from_win(key)
            self._learn_cells(before, action)
        if died or solved:
            return                             # a death or a new level is not a place in this graph
        refilled = value is not None and after_value is not None and after_value > value
        if refilled and ((frame != self.level_start) & ~self._mask()).mean() < self.SAME_START:
            self.gauge.observe_restart(value)  # the game put the player back without a game over
            self.restarting = True
            self.route = []
            return
        spent = 1 if value is None or after_value is None else value - after_value
        self.history.append(Step(before, action, frame, spent))
        node.edges[action] = (target, spent)
        if self.route and target != self._expected:
            self.route = []                    # the walk did not go as known: think again

    def _learn_from_win(self, last: bytes) -> None:
        """The graph of a solved level says how far from the win every state was."""
        if self.progress is None:
            return
        distance = self.graph.distance_to(last)
        self.progress.add_level([(n.frame, None if k not in distance else distance[k] + 1)
                                 for k, n in self.graph.nodes.items()])

    def _learn_cells(self, last: np.ndarray, action: ActionKey) -> None:
        """A level solved by recolouring same-shaped objects shows how those objects should look."""
        if self.cell_learner is None or not isinstance(action, tuple) or not self.history:
            return
        mask = self._mask()
        changed = np.logical_or.reduce([s.before != s.after for s in self.history]) & ~mask
        final = self._after_click(last, action, mask)
        found = objects(final, mask)
        shapes = {t.kind[1] for t in found if changed[t.pixels].all()}
        colours = sorted({int(c) for s in self.history for f in (s.before, s.after) for c in np.unique(f[changed])})
        cells = [t for t in found if t.kind[1] in shapes and t.kind[0] in colours]
        if len(cells) < self.MIN_CELLS or not changed[~np.logical_or.reduce([t.pixels for t in cells])].sum() == 0:
            return                             # something other than whole objects changed colour
        hidden = np.where(mask, HIDDEN, final.astype(np.int16))
        self.cell_learner.add_level(hidden, [(t.pixels, t.box) for t in cells], colours)
        self.cell_learner.train()
        self.cell_shapes |= shapes

    def _after_click(self, last: np.ndarray, action: ActionKey, mask: np.ndarray) -> np.ndarray:
        """The frame a winning click produced is never shown: the next level comes instead."""
        final = last.copy()
        same = next((s for s in reversed(self.history) if s.action == action and (s.before != s.after).any()), None)
        if same is not None:
            moved = (same.before != same.after) & ~mask
            for colour in np.unique(same.before[moved]):       # the same place reacts the same way
                wanted = moved & (same.before == colour)
                final[wanted & (final == colour)] = same.after[wanted][0]
                others = wanted & (final != same.after[wanted][0])
                final[others] = colour
            return final
        changes = {(int(a), int(b)) for s in self.history for a, b in zip(s.before[s.before != s.after],
                                                                          s.after[s.before != s.after])}
        x, y = action
        clicked = next((t for t in objects(last, mask) if t.pixels[y, x]), None)
        if clicked is not None:
            next_colour = next((b for a, b in changes if a == clicked.kind[0]), None)
            if next_colour is not None:
                final[clicked.pixels] = next_colour
        return final

    def _value(self, key: bytes) -> float:
        node = self.graph.nodes[key]
        if node.value_version != self.progress.version:
            node.value = float(self.progress.score([node.frame])[0])
            node.value_version = self.progress.version
        return node.value

    def _best_frontier(self, key: bytes, frame: np.ndarray) -> list[ActionKey] | None:
        """Route to the frontier state that looks closest to a win, walking included."""
        routes = self.graph.distances(key)
        energy = self.gauge.read(frame)
        frontier = [k for k, route in routes.items() if route and (energy is None or len(route) < energy)
                    and any(self.priority(t) >= 0 for t in self.graph.untried(k))]
        if not frontier:
            return None
        return routes[max(frontier, key=lambda k: self._value(k) - self.DETOUR_COST * len(routes[k]))]

    # ----------------------------------------------------------------- deciding
    def priority(self, t: Target) -> float:
        """Higher first. Negative: it never reacted, as this kind of object or at this place."""
        return min(self.kinds.priority(t.kind), self.places.priority(t.action))

    def _decide(self, key: bytes, node: Node, frame: np.ndarray) -> tuple[ActionKey, str]:
        if self.route:
            return self._walk(key, "walking to an untried action")
        click = self._cell_click(node, frame)
        if click is not None:
            self._expected = None
            return click
        untried = self.graph.untried(key)
        guided = self.progress is not None and self.progress.ready
        if guided and (not any(self.priority(t) >= 0 for t in untried) or self._better_elsewhere(key)):
            route = self._best_frontier(key, frame)
            if route:
                self.route = route
                return self._walk(key, "walking to the state that looks closest to a win")
        if untried:
            best = max(untried, key=self.priority)
            if self.priority(best) >= 0:
                self._expected = None
                return best.action, "untried here"
        promising = lambda k: any(self.priority(t) >= 0 for t in self.graph.untried(k))
        energy = self.gauge.read(frame)
        route = self.graph.route(key, promising, energy)
        if route:
            self.route = route
            return self._walk(key, "walking to an untried action")
        if untried:
            self._expected = None
            return max(untried, key=self.priority).action, "nothing better left: long shot"
        route = self.graph.route(key, lambda k: bool(self.graph.untried(k)), None)
        if route:
            self.route = route
            return self._walk(key, "walking to the last untried actions")
        self._expected = None
        fallback = next(iter(node.edges), node.targets[0].action if node.targets else 1)
        return fallback, "everything here was tried"

    def _cell_click(self, node: Node, frame: np.ndarray) -> tuple[ActionKey, str] | None:
        """The cell most likely to be wrong. If all look right and the level goes on, one verdict is mistaken:
        try the cell CellNet is least sure about, and it will put the cell back if that does not win."""
        if self.cell_learner is None or not self.cell_learner.ready:
            return None
        cells = [t for t in node.targets if t.box is not None and t.kind[1] in self.cell_shapes]
        if not cells:
            return None
        hidden = np.where(self._mask(), HIDDEN, frame.astype(np.int16))
        wrong = self.cell_learner.wrong(hidden, [t.box for t in cells], {t.kind[0] for t in cells})
        usable = [(score, t.action) for score, t in zip(wrong, cells)
                  if t.action not in self.dead_cells and self.cell_clicks[t.action] < self.CELL_CLICKS]
        for score, action in sorted(usable, key=lambda c: c[0], reverse=True):
            if score > 0.5:
                self.cell_clicks[action] += 1
                return action, "CellNet: this object looks wrong"
            if action not in self.doubted:
                self.doubted.add(action)
                self.cell_clicks[action] += 1
                return action, "CellNet: all look right, trying the one it is least sure of"
        return None

    def _better_elsewhere(self, key: bytes) -> bool:
        """Leave this state if another frontier state looks better even after walking there."""
        routes = self.graph.distances(key)
        here = self._value(key)
        return any(self._value(k) - self.DETOUR_COST * len(r) > here for k, r in routes.items()
                   if r and any(self.priority(t) >= 0 for t in self.graph.untried(k)))

    def _walk(self, key: bytes, why: str) -> tuple[ActionKey, str]:
        action = self.route.pop(0)
        self._expected = self.graph.nodes[key].edges[action][0] if action in self.graph.nodes[key].edges else None
        return action, why

    @staticmethod
    def _game_action(action: ActionKey) -> GameAction:
        if isinstance(action, tuple):
            game_action = GameAction.from_id(CLICK_ID)
            game_action.set_data({"x": action[0], "y": action[1]})
            return game_action
        return GameAction.from_id(action)

    # ---------------------------------------------------------------- reporting
    def stats(self) -> dict[str, Any]:
        found = self.gauge.gauge
        return {"level_actions": list(self.level_actions), "states": len(self.graph.nodes), "resets": self.resets,
                "kinds": len(self.kinds.tries), "gauge": None if found is None else (found.fill, found.rect),
                "cell levels": len(self.cell_learner.levels) if self.cell_learner else 0,
                "why": dict(self.why_counts)}
