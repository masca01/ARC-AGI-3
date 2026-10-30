"""ls20 specialist: the observation-only causal agent written by a teammate (arc-special-tile-agent/agent.py).

Ported unchanged except for:
  - the standalone CLI (play/main) was removed; CausalAgent in causal_agent.py drives it instead,
  - `locate_player` lets another module (the CNN in control_net.py) replace the hand-made
    player detector; None keeps the original colour-12 detector,
  - `restart_attempt()` tells the agent the environment was RESET after a GAME_OVER, so the
    same level is retried while keeping everything learned about it,
  - `slide_model` (SlideNet in slide_net.py) predicts pushes from how cells look. Planning uses
    those predictions wherever no exact observation exists, and confidently predicted pushers are
    not re-tested. None keeps the original behaviour,
  - `track_objectives` (ObjectiveTracker in objectives.py) keeps track of which doors are still open.
    A door that disappeared after being solved is no longer a goal, so in levels with several doors the
    agent moves on to the next one. False keeps the original behaviour (every door ever seen stays a
    goal, which deadlocks after the first door: the key stays locked to the solved door),
  - limited vision (level 7 shows only a disc around the player): darkness is told apart from walls,
    everything seen is remembered when it goes out of view, the HUD key is read where it was on earlier
    levels, and when no objective is known the agent walks to the edge of the known map to reveal more
    (frontier exploration). On fully visible levels none of this changes anything,
  - `use_memory` (ObjectMemory in memory.py) remembers every significant object of the level: the
    objective and its pattern, transformer tiles, and every refill, including used ones, which come
    back after a lost life. A remembered tile is only forgotten when it vanished from a fully lit cell
    without the player being involved. False keeps the original behaviour (a tile that is not detected
    in view is deleted, so used refills are forgotten for the rest of the level),
  - `careful_exploration`: exploratory moves (reveal, frontier, gate tests, refills, fallback) treat
    known mechanism tiles as obstacles unless they are the target, so exploring never undoes progress
    on the key by accident. Deliberate experiments and the solution plan still use them,
  - `commit_plans`: keep executing the chosen plan while every step turns out as predicted, instead of
    searching again from scratch each step. Replanning every step with a model that changes a little
    each step (moving-tile phases, slide predictions from the current frame) and a capped search made
    consecutive plans point in opposite directions, so the agent stepped back and forth.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from arcengine import GameAction, GameState

from .memory import ObjectMemory
from .objectives import ObjectiveTracker

Pos = Tuple[int, int]
Key = Tuple[int, ...]
DynamicConfig = Tuple[Tuple[str, int, int, int, int], ...]
BG = 5
MOVE = {
    GameAction.ACTION1: (-1, 0),
    GameAction.ACTION2: (1, 0),
    GameAction.ACTION3: (0, -1),
    GameAction.ACTION4: (0, 1),
}
OPPOSITE = {
    GameAction.ACTION1: GameAction.ACTION2,
    GameAction.ACTION2: GameAction.ACTION1,
    GameAction.ACTION3: GameAction.ACTION4,
    GameAction.ACTION4: GameAction.ACTION3,
}


@dataclass
class TileModel:
    signature: str
    votes: Counter = field(default_factory=Counter)
    transitions: Dict[Key, Key] = field(default_factory=dict)
    color_map: Dict[int, int] = field(default_factory=dict)
    shape_map: Dict[Tuple[int, ...], Tuple[int, ...]] = field(default_factory=dict)
    visits: int = 0

    @property
    def role(self) -> str:
        return self.votes.most_common(1)[0][0] if self.votes else "unknown"

    def predict(self, key: Key) -> Optional[Key]:
        if key in self.transitions:
            return self.transitions[key]
        a = np.asarray(key, dtype=np.int16).reshape(3, 3)
        if self.role == "rotate_cw":
            return tuple(map(int, np.rot90(a, -1).ravel()))
        if self.role == "rotate_ccw":
            return tuple(map(int, np.rot90(a, 1).ravel()))
        if self.role == "rotate_180":
            return tuple(map(int, np.rot90(a, 2).ravel()))
        if self.role == "recolor":
            values = [x for x in key if x != BG]
            if not values:
                return None
            old = Counter(values).most_common(1)[0][0]
            new = self.color_map.get(old)
            if new is None:
                return None
            return tuple(new if x != BG else BG for x in key)
        if self.role == "reshape":
            old_mask = tuple(int(x != BG) for x in key)
            new_mask = self.shape_map.get(old_mask)
            if new_mask is None:
                return None
            color = Counter(x for x in key if x != BG).most_common(1)[0][0]
            return tuple(color if occupied else BG for occupied in new_mask)
        return None


@dataclass
class Tile:
    pos: Pos
    signature: str
    kind: str = "mechanism"
    visits: int = 0


@dataclass
class Door:
    pos: Pos
    goal: Key
    box: Optional[Tuple[int, int, int, int]] = None   # y0, x0, y1, x1 of the goal box in pixels


@dataclass(frozen=True)
class Gate:
    signature: str
    orientation: str
    bbox: Tuple[int, int, int, int]
    side_a: Pos
    side_b: Pos


@dataclass
class DynamicTransition:
    source: Pos
    action: int
    destination: Pos
    gate_signature: str
    config_before: DynamicConfig
    config_after: DynamicConfig
    observations: int = 1


@dataclass
class MovingTrack:
    signature: str
    history: List[Tuple[int, Pos]] = field(default_factory=list)
    period: Optional[int] = None
    confidence: float = 0.0
    phase_positions: Dict[int, Counter] = field(default_factory=dict)

    def observe(self, tick: int, pos: Pos) -> None:
        if not self.history or self.history[-1] != (tick, pos):
            self.history.append((tick, pos))
            self.history = self.history[-96:]
        self.fit_period()

    def fit_period(self) -> None:
        if len(self.history) < 6:
            return
        by_tick = dict(self.history)
        best = None
        for period in range(2, min(20, max(by_tick) - min(by_tick) + 1)):
            comparable = matches = 0
            for tick, pos in self.history:
                if tick - period in by_tick:
                    comparable += 1
                    matches += by_tick[tick - period] == pos
            if comparable >= 3:
                score = matches / comparable
                candidate = (score, comparable, -period, period)
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] < 0.75:
            return
        self.confidence, self.period = best[0], best[3]
        phases: Dict[int, Counter] = {}
        for tick, pos in self.history:
            phases.setdefault(tick % self.period, Counter())[pos] += 1
        self.phase_positions = phases

    def predict(self, tick: int) -> Optional[Pos]:
        if self.period is not None:
            votes = self.phase_positions.get(tick % self.period)
            if votes:
                return votes.most_common(1)[0][0]
        if self.history and self.history[-1][0] == tick:
            return self.history[-1][1]
        return None

    def reset_phase(self, visible: Iterable[Pos]) -> None:
        old_period = self.period
        old_phases = self.phase_positions
        visible = list(visible)
        if not visible and old_phases:
            # Out of sight at the restart (limited vision): the restart puts mechanisms back at their
            # start too, so the learned cycle still holds with the tick counter back at 0. Keep it
            # instead of forgetting it. (When the object is in sight it is re-anchored as before.)
            return
        self.history = [(0, visible[0])] if visible else []
        self.confidence = 0.0
        self.period = old_period
        self.phase_positions = {}
        if old_period is None or not visible or not old_phases:
            return
        anchor = None
        for phase, votes in old_phases.items():
            if visible[0] in votes:
                anchor = phase
                break
        if anchor is None:
            return
        for phase, votes in old_phases.items():
            self.phase_positions[(phase - anchor) % old_period] = Counter(votes)
        self.confidence = 0.75


@dataclass
class Scene:
    frames: np.ndarray
    frame: np.ndarray
    level: int
    player: Pos
    origin: Pos
    free: set[Pos]
    visible_tiles: Dict[Pos, Tile]
    doors: List[Door]
    key: Key
    resource: int
    lives: int
    gates: List[Gate]
    dynamic_config: DynamicConfig
    unseen: Optional[np.ndarray] = None                     # (64, 64) bool: darkness, nothing visible
    visible_cells: set = field(default_factory=set)         # grid cells fully in sight right now
    grid_cells: set = field(default_factory=set)            # every cell of the playing area


@dataclass
class Metrics:
    game_id: str
    episode: int
    won: bool
    final_state: str
    levels_completed: int
    actions: int
    interactions: int
    dynamic_transitions_learned: int
    semantic_repeats: int
    models: Dict[str, str]


class LS20ExtendedStateAgent:
    """Observation-only causal agent specialized to the visual grammar of ls20.

    State used by the planner:
      (player, key, resource, consumed pickups, dynamic configuration)

    The agent does not read environment source or hidden attributes.  It learns
    non-local transitions whenever an action moves the player somewhere other
    than the adjacent predicted cell.  Static key-transform models persist
    across levels; position-specific dynamic transitions reset each level.
    """

    def __init__(self, max_actions: int = 1200, seed: int = 0, verbose: bool = True):
        self.max_actions = max_actions
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        self.models: Dict[str, TileModel] = {}
        # frame -> (grid position, grid origin) or None; None means "use detect_player"
        self.locate_player: Optional[Callable[[np.ndarray], Optional[Tuple[Pos, Pos]]]] = None
        # (previous scene, previous action, scene) -> {cell: (class, probability, slide end)}; None = off
        self.slide_model: Optional[Callable] = None
        self.track_objectives = False
        self.use_memory = False
        self.careful_exploration = False
        self.commit_plans = False
        self.committed: Optional[dict] = None
        self.plan_reuses = 0
        self.memory = ObjectMemory()
        self.memory_events: List[Tuple[str, str]] = []
        self.key_box: Optional[Tuple[int, int, int, int]] = None   # HUD key box (x0, y0, w, h), kept across levels
        self.objectives = ObjectiveTracker()
        self.objective_events: List[Tuple[str, str]] = []
        self.reset_episode()

    def reset_episode(self) -> None:
        self.level = -1
        self.actions = 0
        self.interactions = 0
        self.semantic_repeats = 0
        self.last_scene: Optional[Scene] = None
        self.last_action: Optional[GameAction] = None
        self.reason = "initializing"
        self.last_reason = ""
        self.tiles: Dict[Pos, Tile] = {}
        self.doors: Dict[Pos, Door] = {}
        self.free_memory: set[Pos] = set()
        self.seen_cells: set[Pos] = set()           # cells ever visible this level (map memory)
        self.reveal_intent: Optional[Tuple[str, Pos]] = None   # committed exploration target
        self.consumed: set[Pos] = set()
        self.blocked_edges: set[Tuple[Pos, Pos]] = set()
        self.dynamic_edges: Dict[Tuple[Pos, int, DynamicConfig], DynamicTransition] = {}
        self.dynamic_rules: Dict[Tuple[str, int], Counter] = {}
        self.moving_signatures: set[str] = set()
        self.signature_positions: Dict[str, set[Pos]] = {}
        self.moving_tracks: Dict[str, MovingTrack] = {}
        self.current_mobile_tiles: Dict[Pos, Tile] = {}
        self.level_tick = -1
        self.level_signature_visits = Counter()
        self.seen = Counter()
        self.resource_capacity = 42
        self.step_cost = 1
        self.pending_restart = False
        self.last_plan: List[GameAction] = []   # whole plan behind the last action, for replays
        self.slides: Dict[Pos, Tuple[int, float, Optional[Pos]]] = {}

    def restart_attempt(self) -> None:
        """The environment was RESET after a GAME_OVER: the next frame restarts the same level."""
        self.pending_restart = True

    @staticmethod
    def frame_stack(obs) -> np.ndarray:
        a = np.asarray(obs.frame, dtype=np.int16)
        if a.ndim == 2:
            a = a[None, ...]
        while a.ndim > 3:
            a = a[-1]
        return a

    @staticmethod
    def components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        seen = np.zeros(mask.shape, dtype=bool)
        out: List[List[Tuple[int, int]]] = []
        h, w = mask.shape
        for yy, xx in np.argwhere(mask):
            y, x = int(yy), int(xx)
            if seen[y, x]:
                continue
            q = [(y, x)]
            seen[y, x] = True
            points = []
            while q:
                y, x = q.pop()
                points.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            out.append(points)
        return out

    @classmethod
    def detect_player(cls, frame: np.ndarray) -> Tuple[Pos, Pos]:
        candidates = []
        for points in cls.components(frame == 12):
            ys, xs = zip(*points)
            width = max(xs) - min(xs) + 1
            height = max(ys) - min(ys) + 1
            if len(points) >= 8 and width >= 4 and height <= 3 and max(ys) < 55:
                candidates.append(points)
        if not candidates:
            raise RuntimeError("Could not detect the orange-blue player")
        points = max(candidates, key=len)
        ys, xs = zip(*points)
        y0, x0 = min(ys), min(xs)
        origin = (y0 % 5, x0 % 5)
        pos = (int(round((y0 - origin[0]) / 5)), int(round((x0 - origin[1]) / 5)))
        return pos, origin

    @classmethod
    def extract_symbols(cls, frame: np.ndarray):
        """(carried key or None, door goals, key box (x0, y0, w, h) or None)."""
        boxes = []
        for points in cls.components(frame == BG):
            if len(points) < 20:
                continue
            ys, xs = zip(*points)
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        current, key_box = None, None
        goals = []
        for x0, y0, x1, y1 in boxes:
            width, height = x1 - x0 + 1, y1 - y0 + 1
            if width >= 9 and height >= 9 and y0 >= 50:
                patch = frame[y0 + 2:y0 + 8, x0 + 2:x0 + 8]
                if patch.shape == (6, 6):
                    current = tuple(map(int, patch.reshape(3, 2, 3, 2)[:, 0, :, 0].ravel()))
                    key_box = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
            elif width == 7 and height == 7:
                patch = frame[y0 + 2:y0 + 5, x0 + 2:x0 + 5]
                if patch.shape == (3, 3):
                    goals.append(((x0, y0, x1, y1), tuple(map(int, patch.ravel()))))
        return current, goals, key_box

    @staticmethod
    def tile_signature(patch: np.ndarray) -> str:
        a = patch.copy()
        a[a == 3] = BG
        return hashlib.sha1(a.astype(np.int8).tobytes()).hexdigest()[:12]

    @staticmethod
    def resource(frame: np.ndarray) -> int:
        return int(np.sum(frame[61:63, 13:55] == 11) // 2)

    @staticmethod
    def lives(frame: np.ndarray) -> int:
        return int(np.sum(frame[61:63, 56:64] == 8) // 4)

    @classmethod
    def detect_gates(cls, frame: np.ndarray, origin: Pos) -> List[Gate]:
        """Detect isolated one-cell-thick line objects on grid boundaries.

        No semantic direction is assumed.  Both incident crossings remain
        candidates until an observed action establishes their transition.
        """
        oy, ox = origin
        gates = []
        for points in cls.components(frame == 1):
            if not 3 <= len(points) <= 7:
                continue
            ys, xs = zip(*points)
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            if y1 >= 53:
                continue
            if y0 == y1 and x1 - x0 >= 3:
                c = int(round((x0 - ox) / 5))
                upper = int(np.floor((y0 - oy) / 5))
                a, b, orientation = (upper, c), (upper + 1, c), "horizontal"
            elif x0 == x1 and y1 - y0 >= 3:
                r = int(round((y0 - oy) / 5))
                left = int(np.floor((x0 - ox) / 5))
                a, b, orientation = (r, left), (r, left + 1), "vertical"
            else:
                continue
            local = frame[max(0, y0 - 2):min(53, y1 + 3), max(0, x0 - 2):min(64, x1 + 3)]
            signature = orientation[0] + hashlib.sha1(local.astype(np.int8).tobytes()).hexdigest()[:10]
            gates.append(Gate(signature, orientation, (x0, y0, x1, y1), a, b))
        return gates

    @classmethod
    def dynamic_configuration(cls, frame: np.ndarray, gates: Iterable[Gate]) -> DynamicConfig:
        config = [(g.orientation, *g.bbox) for g in gates]
        # Large compact color-4 components are included as possible moving blocks.
        for points in cls.components(frame[:53] == 4):
            if not 4 <= len(points) <= 400:
                continue
            ys, xs = zip(*points)
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            if (x1 - x0 >= 3 or y1 - y0 >= 3) and x1 - x0 < 30 and y1 - y0 < 30:
                config.append(("block", x0, y0, x1, y1))
        return tuple(sorted(config))

    def parse(self, obs) -> Scene:
        frames = self.frame_stack(obs)
        frame = frames[-1]
        level = int(obs.levels_completed or 0)
        if self.locate_player is None:
            player, origin = self.detect_player(frame)
        else:
            located = self.locate_player(frame)
            if located is None:
                raise RuntimeError("player locator found no player")
            player, origin = located
        key, raw_goals, key_box = self.extract_symbols(frame)
        if self.key_box is None and key_box is not None:
            self.key_box = key_box
        if self.key_box is not None and key_box != self.key_box:
            # HUD elements do not move. A box elsewhere or of another size is the real box merged with
            # darkness (limited vision): read the key where the box was found cleanly before.
            x0, y0 = self.key_box[:2]
            patch = frame[y0 + 2:y0 + 8, x0 + 2:x0 + 8]
            if patch.shape == (6, 6):
                key = tuple(map(int, patch.reshape(3, 2, 3, 2)[:, 0, :, 0].ravel()))
        if key is None:
            raise RuntimeError("Could not extract the current 3x3 key")
        unseen = self.darkness(frame)
        oy, ox = origin
        doors = []
        for (x0, y0, x1, y1), goal in raw_goals:
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            pos = (int(round((cy - oy) / 5)), int(round((cx - ox) / 5)))
            doors.append(Door(pos, goal, (y0, x0, y1, x1)))
        door_positions = {d.pos for d in doors}
        free: set[Pos] = set()
        tiles: Dict[Pos, Tile] = {}
        visible_cells: set[Pos] = set()
        grid_cells: set[Pos] = set()
        special_colors = {0, 1, 2, 6, 7, 8, 10, 11, 13, 14, 15}
        for r, y in enumerate(range(oy, min(60, frame.shape[0] - 4), 5)):
            for c, x in enumerate(range(ox, min(60, frame.shape[1] - 4), 5)):
                patch = frame[y:y + 5, x:x + 5]
                hist = Counter(map(int, patch.ravel()))
                pos = (r, c)
                if y < 53:
                    grid_cells.add(pos)
                in_sight = not unseen[y:y + 5, x:x + 5].any()     # fully lit: the only cells we judge
                if in_sight:
                    visible_cells.add(pos)
                floor = hist[3] >= 8
                is_player = hist[12] >= 8 if self.locate_player is None else pos == player
                pickup = hist[11] >= 8
                if floor or is_player or pos in door_positions:
                    free.add(pos)
                salience = sum(hist[k] for k in special_colors)
                if y < 53 and in_sight and (floor or pickup) and salience >= 2 and not is_player and pos not in door_positions:
                    tiles[pos] = Tile(pos, self.tile_signature(patch), "pickup_visual" if pickup else "mechanism")
        free.add(player)
        gates = self.detect_gates(frame, origin)
        visible_cells.add(player)
        return Scene(
            frames, frame, level, player, origin, free, tiles, doors, key,
            self.resource(frame), self.lives(frame), gates,
            self.dynamic_configuration(frame, gates), unseen, visible_cells, grid_cells,
        )

    @classmethod
    def darkness(cls, frame: np.ndarray) -> np.ndarray:
        """Pixels that show nothing: background-coloured regions connected to the screen border.

        Door and key boxes use the same colour but are enclosed, so they are not darkness. On fully
        visible levels this only covers the margin and the HUD background, outside the playing area.
        """
        dark = np.zeros(frame.shape, bool)
        h, w = frame.shape
        for points in cls.components(frame == BG):
            ys, xs = zip(*points)
            if min(ys) == 0 or min(xs) == 0 or max(ys) == h - 1 or max(xs) == w - 1:
                dark[tuple(np.array(points).T)] = True
        return dark

    @staticmethod
    def key_array(key: Key) -> np.ndarray:
        return np.asarray(key, dtype=np.int16).reshape(3, 3)

    @classmethod
    def classify_key_transition(cls, before: Key, after: Key) -> str:
        if before == after:
            return "none"
        a, b = cls.key_array(before), cls.key_array(after)
        if np.array_equal(b, np.rot90(a, -1)):
            return "rotate_cw"
        if np.array_equal(b, np.rot90(a, 1)):
            return "rotate_ccw"
        if np.array_equal(b, np.rot90(a, 2)):
            return "rotate_180"
        if np.array_equal(a != BG, b != BG):
            return "recolor"
        return "reshape"

    @staticmethod
    def dominant_color(key: Key) -> int:
        values = [x for x in key if x != BG]
        return Counter(values).most_common(1)[0][0] if values else BG

    def begin_level(self, scene: Scene) -> None:
        self.level = scene.level
        self.tiles = {}
        self.doors = {}
        self.objectives.reset_level()
        self.memory.reset_level()
        self.slides = {}
        self.free_memory = set()
        self.consumed = set()
        self.blocked_edges = set()
        self.dynamic_edges = {}
        self.dynamic_rules = {}
        self.seen_cells = set()
        self.reveal_intent = None
        self.moving_signatures = set()
        self.signature_positions = {}
        self.moving_tracks = {}
        self.current_mobile_tiles = {}
        self.level_tick = -1
        self.level_signature_visits = Counter()
        self.step_cost = 1
        self.resource_capacity = max(42, scene.resource)
        if self.verbose:
            print(f"\n--- entered level {scene.level + 1}: key={scene.key}, goals={[d.goal for d in scene.doors]} ---")

    def merge_scene(self, scene: Scene) -> None:
        self.level_tick += 1
        self.free_memory |= scene.free
        self.seen_cells |= scene.visible_cells
        visible = scene.visible_cells
        self.memory_events = []
        if self.use_memory:
            detections = {}
            for pos, tile in scene.visible_tiles.items():
                if tile.signature in self.moving_signatures:
                    continue                                  # moving tiles are tracked, not placed
                model = self.models.get(tile.signature)
                refill = tile.kind == "pickup_visual" or (model is not None and model.role == "resource")
                detections[pos] = ("refill" if refill else "tile", tile.signature,
                                   model.role if model is not None else None)
            for door in scene.doors:
                detections[door.pos] = ("goal", "goal", door.goal)
            previous_player = self.last_scene.player if self.last_scene is not None else None
            self.memory_events = self.memory.observe(self.level_tick, detections, scene.visible_cells,
                                                     scene.player, previous_player)
        current_positions: Dict[str, set[Pos]] = {}
        by_signature: Dict[str, Dict[Pos, Tile]] = {}
        for pos, tile in scene.visible_tiles.items():
            current_positions.setdefault(tile.signature, set()).add(pos)
            by_signature.setdefault(tile.signature, {})[pos] = tile

        all_signatures = set(self.signature_positions) | set(current_positions)
        for signature in all_signatures:
            positions = current_positions.get(signature, set())
            remembered = self.signature_positions.get(signature, set())
            # out of sight is not gone: only judge the positions that can be seen now
            hidden = {p for p in remembered if p not in visible}
            previous = remembered - hidden
            track = self.moving_tracks.setdefault(signature, MovingTrack(signature))
            if len(positions) == 1:
                track.observe(self.level_tick, next(iter(positions)))
            if previous and positions and previous != positions and len(previous) == len(positions) == 1:
                (a, b), (c, d) = next(iter(previous)), next(iter(positions))
                # moving tiles step a cell at a time; the same icon far away is another copy
                # (with limited vision the two copies are never in sight together)
                if abs(a - c) + abs(b - d) <= 2:
                    self.moving_signatures.add(signature)
            if signature in self.moving_signatures:
                for pos in list(self.tiles):
                    if self.tiles[pos].signature == signature:
                        del self.tiles[pos]
            else:
                for stale in previous - positions:
                    if stale in self.tiles and self.tiles[stale].signature == signature:
                        item = self.memory.items.get(stale) if self.use_memory else None
                        if item is not None and item.signature == signature:
                            if item.status == "used":
                                self.consumed.add(stale)      # used, not gone: it comes back after a restart
                            continue                          # remembered (under the player, or used)
                        del self.tiles[stale]
            self.signature_positions[signature] = set(positions) | (hidden if signature not in self.moving_signatures else set())

        self.current_mobile_tiles = {}
        for pos, tile in scene.visible_tiles.items():
            if tile.signature in self.moving_signatures:
                self.current_mobile_tiles[pos] = tile
                continue
            old = self.tiles.get(pos)
            if old is None or old.signature != tile.signature:
                self.tiles[pos] = tile
            else:
                old.kind = tile.kind
        for door in scene.doors:
            self.doors[door.pos] = door
        self.objective_events = []
        if self.track_objectives:
            seen_doors = {door.pos: (door.pos, door.box, door.goal) for door in scene.doors}
            py, px = scene.origin[0] + 5 * scene.player[0], scene.origin[1] + 5 * scene.player[1]
            self.objective_events = self.objectives.update(self.level_tick, seen_doors, scene.player,
                                                           scene.frame, (py, px, py + 4, px + 4),
                                                           None if scene.unseen is None else ~scene.unseen)
            if self.doors and not self.objectives.open():
                # every goal looks achieved but the level did not end: the tracker was wrong somewhere
                for objective in self.objectives.objectives.values():
                    objective.status = "open"
                self.objective_events.append(("reopened", "all objectives looked done but the level "
                                                          "did not end: reopening them"))

    def open_doors(self) -> Dict[Pos, Door]:
        """Doors that are still goals (all remembered doors when objective tracking is off)."""
        if not self.track_objectives:
            return self.doors
        return {pos: door for pos, door in self.doors.items() if self.objectives.is_open(pos)}

    def temporal_cycle(self) -> int:
        periods = [
            track.period for signature, track in self.moving_tracks.items()
            if signature in self.moving_signatures and track.period
        ]
        cycle = 1
        for period in periods:
            cycle = math.lcm(cycle, int(period))
            if cycle > 120:
                return 120
        return cycle

    def mobile_signatures_at(self, pos: Pos, tick: int) -> List[str]:
        result = []
        for signature in self.moving_signatures:
            track = self.moving_tracks.get(signature)
            if track is not None and track.predict(tick) == pos:
                result.append(signature)
        return result

    def mobile_contact_tile(self, scene: Scene) -> Optional[Tile]:
        candidates = self.mobile_signatures_at(scene.player, self.level_tick)
        if not candidates:
            return None
        signature = max(candidates, key=lambda sig: self.models.get(sig, TileModel(sig)).visits)
        return Tile(scene.player, signature, "mechanism")

    def reset_temporal_phase(self, scene: Scene) -> None:
        self.level_tick = 0
        for signature, track in self.moving_tracks.items():
            positions = [
                pos for pos, tile in scene.visible_tiles.items()
                if tile.signature == signature
            ]
            track.reset_phase(positions)

    def learn_tile(self, tile: Tile, before: Key, after: Key, resource_before: int, resource_after: int) -> None:
        model = self.models.setdefault(tile.signature, TileModel(tile.signature))
        role = self.classify_key_transition(before, after)
        if tile.kind == "pickup_visual" and before == after:
            role = "resource"
        if role == "none" and model.role != "unknown":
            role = model.role
        model.visits += 1
        model.votes[role] += 1
        model.transitions[before] = after
        if role == "recolor":
            model.color_map[self.dominant_color(before)] = self.dominant_color(after)
        if role == "reshape":
            before_mask = tuple(int(x != BG) for x in before)
            after_mask = tuple(int(x != BG) for x in after)
            a = np.asarray(before_mask, dtype=np.uint8).reshape(3, 3)
            b = np.asarray(after_mask, dtype=np.uint8).reshape(3, 3)
            for rotation in range(4):
                source = tuple(map(int, np.rot90(a, rotation).ravel()))
                destination = tuple(map(int, np.rot90(b, rotation).ravel()))
                model.shape_map[source] = destination
        self.level_signature_visits[tile.signature] += 1
        if role == "resource":
            self.consumed.add(tile.pos)
        tile.visits += 1
        self.interactions += 1
        if self.verbose:
            print(f"  learned tile={tile.pos} role={role} key_changed={before != after} resource={resource_before}->{resource_after}")

    @staticmethod
    def attempted_destination(pos: Pos, action: GameAction) -> Pos:
        dr, dc = MOVE[action]
        return pos[0] + dr, pos[1] + dc

    def crossed_gate(self, source: Pos, attempted: Pos, gates: Iterable[Gate]) -> Optional[Gate]:
        for gate in gates:
            if {source, attempted} == {gate.side_a, gate.side_b}:
                return gate
        return None

    def learn_dynamic_transition(self, previous: Scene, current: Scene, action: GameAction) -> None:
        attempted = self.attempted_destination(previous.player, action)
        if current.player == attempted:
            return
        gate = self.crossed_gate(previous.player, attempted, previous.gates)
        gate_signature = gate.signature if gate else "unassigned"
        if current.player == previous.player:
            self.blocked_edges.add((previous.player, attempted))
            return
        key = (previous.player, action.value, previous.dynamic_config)
        old = self.dynamic_edges.get(key)
        if old and old.destination == current.player:
            old.observations += 1
        else:
            self.dynamic_edges[key] = DynamicTransition(
                previous.player, action.value, current.player, gate_signature,
                previous.dynamic_config, current.dynamic_config,
            )
        relative = (current.player[0] - previous.player[0], current.player[1] - previous.player[1])
        rule_key = (gate_signature, action.value)
        self.dynamic_rules.setdefault(rule_key, Counter())[relative] += 1
        self.free_memory.add(current.player)
        if self.verbose:
            print(f"  learned dynamic transition {previous.player} --{action.name}--> {current.player} gate={gate_signature}")

    def observe_transition(self, scene: Scene) -> None:
        if self.last_scene is None or self.last_action is None:
            return
        previous = self.last_scene
        if scene.level > previous.level:
            return
        reset = self.pending_restart or scene.lives < previous.lives or (
            scene.resource > previous.resource + 10
            and abs(scene.player[0] - previous.player[0]) + abs(scene.player[1] - previous.player[1]) > 2
        )
        if reset:
            self.pending_restart = False
            self.consumed.clear()
            if self.use_memory:
                self.memory.restart(self.level_tick)
            self.blocked_edges.clear()
            self.dynamic_edges = {
                key: edge for key, edge in self.dynamic_edges.items()
                if edge.gate_signature != "unassigned" or edge.observations >= 2
            }
            self.reset_temporal_phase(scene)
            if self.verbose:
                print("  life reset detected; reset dynamic phase, retained causal mechanics")
            return
        delta_resource = previous.resource - scene.resource
        if delta_resource > 0:
            self.step_cost = max(self.step_cost, delta_resource)
        attempted = self.attempted_destination(previous.player, self.last_action)
        self.learn_dynamic_transition(previous, scene, self.last_action)

        tile = None
        if previous.key != scene.key:
            tile = self.mobile_contact_tile(scene)
        if tile is None:
            tile = previous.visible_tiles.get(attempted)
        if tile is None and previous.key != scene.key:
            nearby = []
            for source in (previous.visible_tiles, scene.visible_tiles):
                for pos, candidate in source.items():
                    distance = min(
                        abs(pos[0] - attempted[0]) + abs(pos[1] - attempted[1]),
                        abs(pos[0] - previous.player[0]) + abs(pos[1] - previous.player[1]),
                        abs(pos[0] - scene.player[0]) + abs(pos[1] - scene.player[1]),
                    )
                    if distance <= 1:
                        priority = 0 if candidate.signature in self.moving_signatures else 1
                        nearby.append((priority, distance, candidate))
            if nearby:
                tile = min(nearby, key=lambda item: item[:2])[2]
        contacted = scene.player == attempted or previous.key != scene.key
        resource_gain = scene.resource > previous.resource + self.step_cost
        if tile is not None and (contacted or resource_gain):
            self.learn_tile(tile, previous.key, scene.key, previous.resource, scene.resource)

    SLIDE_CONFIDENCE = 0.5     # use a predicted push in planning above this probability
    SLIDE_SURE = 0.9           # do not re-test a crossing whose outcome is predicted this surely

    def transition_position(self, pos: Pos, action: GameAction,
                            config: DynamicConfig) -> Optional[Tuple[Pos, DynamicConfig]]:
        known = self.dynamic_edges.get((pos, action.value, config))
        if known is not None:
            return known.destination, known.config_after
        nxt = self.attempted_destination(pos, action)
        if (pos, nxt) in self.blocked_edges:
            return None
        slide = self.slides.get(nxt)
        if slide is not None and slide[0] and slide[1] >= self.SLIDE_CONFIDENCE and slide[2] is not None:
            return slide[2], config
        return nxt, config

    def apply_tile(self, pos: Pos, key: Key, resource: int, mask: int,
                   pickup_index: Dict[Pos, int]) -> Tuple[Key, int, int, bool]:
        tile = self.tiles.get(pos)
        if tile is None or tile.signature in self.moving_signatures:
            return key, resource, mask, True
        model = self.models.get(tile.signature)
        bit = (1 << pickup_index[pos]) if pos in pickup_index else 0
        if tile.kind == "pickup_visual" or (model and model.role == "resource"):
            if bit and not mask & bit:
                return key, self.resource_capacity, mask | bit, True
            return key, resource, mask, True
        if model is None or model.role in ("unknown", "none"):
            return key, resource, mask, False
        predicted = model.predict(key)
        return (predicted, resource, mask, True) if predicted is not None else (key, resource, mask, False)

    def apply_mobile_tiles(self, pos: Pos, tick: int, key: Key,
                           target_signature: Optional[str] = None) -> Tuple[Key, bool, bool]:
        contacted_target = False
        result = key
        for signature in self.mobile_signatures_at(pos, tick):
            contacted_target |= signature == target_signature
            model = self.models.get(signature)
            if model is None or model.role in ("unknown", "none"):
                if signature == target_signature:
                    continue
                return result, False, contacted_target
            predicted = model.predict(result)
            if predicted is None:
                if signature == target_signature:
                    continue
                return result, False, contacted_target
            result = predicted
        return result, True, contacted_target

    def mechanism_cells(self) -> set:
        """Cells of known tiles that change the key (anything that is not a refill)."""
        cells = set()
        for pos, tile in self.tiles.items():
            model = self.models.get(tile.signature)
            if tile.kind != "pickup_visual" and not (model and model.role == "resource"):
                cells.add(pos)
        return cells

    def careful_plan(self, scene: Scene, target: Pos, max_nodes: int = 30000,
                     fallback: bool = True) -> Optional[List[GameAction]]:
        """An exploratory path that steps on no mechanism (except the target), if one exists."""
        if not self.careful_exploration:
            return self.joint_plan(scene, target, False, max_nodes)
        path = self.joint_plan(scene, target, False, max_nodes, avoid=self.mechanism_cells() - {target})
        if path is None and fallback:
            path = self.joint_plan(scene, target, False, max_nodes)
        return path

    def joint_plan(self, scene: Scene, target: Optional[Pos] = None,
                   allow_unknown_target: bool = False, max_nodes: int = 140000,
                   mobile_target_signature: Optional[str] = None,
                   avoid: Optional[set] = None) -> Optional[List[GameAction]]:
        free = set(self.free_memory)
        free.add(scene.player)
        free.update(self.doors)
        free.update(t.destination for t in self.dynamic_edges.values())
        for track in self.moving_tracks.values():
            free.update(pos for _, pos in track.history)
        pickups = [
            pos for pos, tile in self.tiles.items()
            if tile.kind == "pickup_visual"
            or (self.models.get(tile.signature) and self.models[tile.signature].role == "resource")
        ]
        pickup_index = {pos: i for i, pos in enumerate(pickups)}
        start_mask = sum(1 << pickup_index[p] for p in self.consumed if p in pickup_index)
        cycle = self.temporal_cycle()
        start_phase = self.level_tick % cycle
        start = (scene.player, scene.key, scene.resource, start_mask, scene.dynamic_config, start_phase)
        queue = deque([start])
        previous = {start: None}
        previous_action = {}
        open_doors = self.open_doors()
        goal_keys = {door.goal for door in open_doors.values()}
        nodes = 0
        while queue and nodes < max_nodes:
            state = queue.popleft()
            nodes += 1
            pos, key, resource, mask, config, phase = state
            if target is not None and pos == target:
                return self.unroll(state, previous, previous_action)
            for action in MOVE:
                transition = self.transition_position(pos, action, config)
                if transition is None:
                    continue
                nxt, next_config = transition
                if nxt not in free or (avoid and nxt in avoid):
                    continue
                if resource < self.step_cost and nxt not in pickups:
                    continue
                next_phase = (phase + 1) % cycle
                new_resource = resource - self.step_cost
                new_key, new_mask = key, mask
                if nxt in self.tiles and not (allow_unknown_target and nxt == target):
                    new_key, new_resource, new_mask, known = self.apply_tile(
                        nxt, new_key, new_resource, new_mask, pickup_index
                    )
                    if not known:
                        continue
                new_key, mobile_known, contacted_target = self.apply_mobile_tiles(
                    nxt, next_phase, new_key, mobile_target_signature
                )
                if not mobile_known:
                    continue
                if key in goal_keys and new_key != key:
                    continue
                door = open_doors.get(nxt)
                if door and door.goal != new_key:
                    continue
                new_state = (nxt, new_key, new_resource, new_mask, next_config, next_phase)
                if new_state not in previous:
                    previous[new_state] = state
                    previous_action[new_state] = action
                    if mobile_target_signature is not None and contacted_target:
                        return self.unroll(new_state, previous, previous_action)
                    if target is None and door and door.goal == new_key:
                        return self.unroll(new_state, previous, previous_action)
                    queue.append(new_state)
        return None

    @staticmethod
    def unroll(state, previous, previous_action) -> List[GameAction]:
        actions = []
        while previous[state] is not None:
            actions.append(previous_action[state])
            state = previous[state]
        return actions[::-1]

    @staticmethod
    def shape_mask(key: Key) -> Tuple[int, ...]:
        return tuple(int(x != BG) for x in key)

    @classmethod
    def canonical_shape(cls, key: Key) -> Tuple[int, ...]:
        mask = np.asarray(cls.shape_mask(key), dtype=np.uint8).reshape(3, 3)
        return min(tuple(map(int, np.rot90(mask, k).ravel())) for k in range(4))

    def goal_discrepancy(self, key: Key, goal: Key) -> Tuple[int, int, int]:
        return (
            int(self.canonical_shape(key) != self.canonical_shape(goal)),
            int(self.dominant_color(key) != self.dominant_color(goal)),
            int(key != goal),
        )

    def required_role(self, scene: Scene) -> Optional[str]:
        doors = self.open_doors()
        if not doors:
            return None
        # objectives that did not react to repeated visits come last, then the closest key match
        goal = min(doors.values(), key=lambda door: (self.objectives.penalty(door.pos) if self.track_objectives else 0,
                                                    self.goal_discrepancy(scene.key, door.goal))).goal
        if scene.key == goal:
            return "door"
        if self.canonical_shape(scene.key) != self.canonical_shape(goal):
            return "reshape"
        if self.dominant_color(scene.key) != self.dominant_color(goal):
            return "recolor"
        return "rotate"

    def model_matches_role(self, model: Optional[TileModel], role: str) -> bool:
        if model is None:
            return False
        if role == "rotate":
            return model.role in ("rotate_cw", "rotate_ccw", "rotate_180")
        return model.role == role

    def interaction_plan(self, scene: Scene, tile: Tile,
                         max_nodes: int = 70000) -> Optional[List[GameAction]]:
        path = self.joint_plan(scene, tile.pos, True, max_nodes)
        if path:
            return path
        if scene.player != tile.pos:
            return None
        for action, reverse in OPPOSITE.items():
            neighbor = self.attempted_destination(scene.player, action)
            if neighbor in self.free_memory and (scene.player, neighbor) not in self.blocked_edges:
                return [action, reverse]
        return None

    def resource_plan(self, scene: Scene) -> Optional[List[GameAction]]:
        options = []
        for tile in self.tiles.values():
            model = self.models.get(tile.signature)
            is_resource = tile.kind == "pickup_visual" or (model and model.role == "resource")
            if not is_resource or tile.pos in self.consumed:
                continue
            path = self.careful_plan(scene, tile.pos, 70000)
            if path is not None:
                options.append(path)
        return min(options, key=len) if options else None

    def semantic_experiment(self, scene: Scene) -> Optional[Tuple[List[GameAction], str]]:
        role = self.required_role(scene)
        if role in (None, "door"):
            return None
        candidates = []
        for tile in self.tiles.values():
            if tile.pos in self.consumed:
                continue
            model = self.models.get(tile.signature)
            if self.model_matches_role(model, role):
                prediction_unknown = model.predict(scene.key) is None
                path = self.interaction_plan(scene, tile)
                if path:
                    candidates.append((not prediction_unknown, len(path), tile.visits, path, f"{tile.pos}"))
        for signature in self.moving_signatures:
            model = self.models.get(signature)
            track = self.moving_tracks.get(signature)
            if not self.model_matches_role(model, role) or track is None or track.period is None:
                continue
            path = self.joint_plan(
                scene, allow_unknown_target=True, max_nodes=90000,
                mobile_target_signature=signature
            )
            if path:
                prediction_unknown = model.predict(scene.key) is None
                candidates.append((not prediction_unknown, len(path), model.visits, path, f"mobile:{signature}"))
        if not candidates:
            return None
        _, _, _, path, label = min(candidates, key=lambda item: item[:3])
        reserve = self.resource_plan(scene)
        safety = self.step_cost * (len(path) + 4)
        if scene.resource < safety and reserve:
            return reserve, "recharge-before-experiment"
        return path, f"active-test:{role}@{label}"

    def discovery_candidates(self, scene: Scene) -> List[Tile]:
        role = self.required_role(scene)
        candidates = []
        for tile in self.tiles.values():
            if tile.pos in self.consumed:
                continue
            model = self.models.get(tile.signature)
            unseen_here = self.level_signature_visits[tile.signature] == 0
            unknown = model is None or model.role in ("unknown", "none")
            context_unknown = (
                model is not None
                and self.model_matches_role(model, role or "")
                and model.predict(scene.key) is None
            )
            if unknown or context_unknown or (tile.kind == "pickup_visual" and unseen_here):
                candidates.append(tile)
        return sorted(candidates, key=lambda tile: (
            0 if tile.kind == "pickup_visual" else 1,
            tile.visits,
            abs(tile.pos[0] - scene.player[0]) + abs(tile.pos[1] - scene.player[1]),
        ))

    def unknown_tiles(self, scene: Scene) -> List[Tile]:
        return self.discovery_candidates(scene)

    def unknown_gate_crossings(self, scene: Scene) -> List[Tuple[Pos, GameAction]]:
        candidates = []
        for gate in scene.gates:
            pairs = [
                (gate.side_a, self.action_between(gate.side_a, gate.side_b)),
                (gate.side_b, self.action_between(gate.side_b, gate.side_a)),
            ]
            for source, action in pairs:
                if action is None or source not in self.free_memory:
                    continue
                learned_here = any(k[0] == source and k[1] == action.value for k in self.dynamic_edges)
                predicted = self.slides.get(self.attempted_destination(source, action))
                learned_here |= predicted is not None and predicted[1] >= self.SLIDE_SURE
                if not learned_here and (source, self.attempted_destination(source, action)) not in self.blocked_edges:
                    candidates.append((source, action))
        return candidates

    @staticmethod
    def action_between(a: Pos, b: Pos) -> Optional[GameAction]:
        delta = b[0] - a[0], b[1] - a[1]
        for action, move in MOVE.items():
            if move == delta:
                return action
        return None

    def plan_to_test_gate(self, scene: Scene, source: Pos, crossing: GameAction) -> Optional[List[GameAction]]:
        path = self.careful_plan(scene, source, 60000, fallback=False)
        if path is None:
            return None
        return path + [crossing]

    def frontier(self, scene: Scene) -> List[Pos]:
        """Known free cells next to cells never seen: walking there reveals more of the map."""
        unknown = scene.grid_cells - self.seen_cells
        if not unknown:
            return []
        cells = []
        for pos in self.free_memory:
            if any((pos[0] + dr, pos[1] + dc) in unknown for dr, dc in MOVE.values()):
                cells.append(pos)
        return sorted(cells, key=lambda p: abs(p[0] - scene.player[0]) + abs(p[1] - scene.player[1]))

    def budget_plan(self, scene: Scene) -> Optional[Tuple[List[GameAction], str]]:
        """The key fits an open door but the resource is too short to get there: refill first.

        Only used when resource really is the blocker: a plan to the door exists with a full budget.
        The refill target is kept (commitment) until it is reached or used up.
        """
        if self.required_role(scene) != "door":
            return None
        intent = self.reveal_intent
        if intent is not None and intent[0] == "refill" and intent[1] != scene.player \
                and intent[1] not in self.consumed and intent[1] in self.tiles:
            path = self.joint_plan(scene, intent[1], False, 30000)
            if path:
                return path, "recharge-before-door"
        if self.joint_plan(dataclasses.replace(scene, resource=10 ** 6)) is None:
            return None                                   # not a budget problem
        reserve = self.resource_plan(scene)
        if reserve is None:
            return None
        self.reveal_intent = ("refill", self.plan_end(scene, reserve))
        return reserve, "recharge-before-door"

    def plan_end(self, scene: Scene, path: List[GameAction]) -> Pos:
        pos, config = scene.player, scene.dynamic_config
        for action in path:
            nxt = self.transition_position(pos, action, config)
            if nxt is None:
                break
            pos, config = nxt
        return pos

    def reveal_plan(self, scene: Scene) -> Optional[Tuple[List[GameAction], str]]:
        """Go to the nearest reachable frontier cell (limited vision: discover the objective).

        Commitment: once a frontier cell or a refill was chosen, keep going there until it is reached
        or no longer useful. Re-deciding every step made the agent ping-pong between "reveal" and
        "recharge first" (each step flips the resource check) across a shape tile.
        """
        intent = self.reveal_intent
        if intent is not None:
            kind, target = intent
            useful = (target != scene.player and
                      (kind == "recharge" and target not in self.consumed and target in self.tiles
                       or kind == "reveal" and target in self.frontier(scene)))
            path = self.careful_plan(scene, target) if useful else None
            if path:
                return path, "recharge-before-reveal" if kind == "recharge" else f"explore:reveal@{target}"
            self.reveal_intent = None
        for cell in self.frontier(scene)[:8]:
            if cell == scene.player:
                continue
            path = self.careful_plan(scene, cell)
            if path:
                reserve = self.resource_plan(scene)
                if scene.resource < self.step_cost * (len(path) + 4) and reserve:
                    self.reveal_intent = ("recharge", self.plan_end(scene, reserve))
                    return reserve, "recharge-before-reveal"
                self.reveal_intent = ("reveal", cell)
                return path, f"explore:reveal@{cell}"
        return None

    def commit(self, scene: Scene, plan: List[GameAction], reason: str) -> None:
        """Remember the plan and what its first step should lead to, to check it next step."""
        self.committed = None
        if len(plan) < 2 or reason == "fallback":
            return
        nxt = self.transition_position(scene.player, plan[0], scene.dynamic_config)
        if nxt is None:
            return
        cell = nxt[0]
        # the key may legitimately change where a tile or a moving tile is expected
        changes_key = cell in self.tiles or bool(self.mobile_signatures_at(cell, self.level_tick + 1))
        self.committed = dict(level=scene.level, rest=list(plan[1:]), reason=reason, expect_pos=cell,
                              expect_key=None if changes_key else scene.key, model=self.model_version())

    def model_version(self) -> tuple:
        """Changes whenever what the agent knows changes: then a committed plan is out of date."""
        return (self.step_cost, len(self.dynamic_edges), len(self.blocked_edges), frozenset(self.tiles),
                tuple(sorted((sig, m.role) for sig, m in self.models.items())), frozenset(self.consumed),
                frozenset(self.moving_signatures), tuple(sorted(self.open_doors())))

    def continue_plan(self, scene: Scene, restarting: bool) -> Optional[Tuple[List[GameAction], str]]:
        """The committed plan, if the last step went as predicted and the rest is still possible."""
        c, self.committed = self.committed, None
        if c is None or restarting or c["level"] != scene.level:
            return None
        if scene.player != c["expect_pos"] or (c["expect_key"] is not None and scene.key != c["expect_key"]):
            return None                                   # surprise: the model was wrong, replan
        if c["model"] != self.model_version():
            return None                                   # learned something since the plan was made
        if c["reason"] != "execute:temporal-state-plan" and self.joint_plan(scene):
            return None                                   # a full solution exists now: take it
        free = set(self.free_memory) | set(self.doors) | {scene.player}
        refills = {pos for pos, tile in self.tiles.items() if pos not in self.consumed and (
            tile.kind == "pickup_visual" or (self.models.get(tile.signature) and self.models[tile.signature].role == "resource"))}
        pos, config, resource = scene.player, scene.dynamic_config, scene.resource
        for action in c["rest"]:
            nxt = self.transition_position(pos, action, config)
            if nxt is None or nxt[0] not in free:
                return None                               # the rest of the plan is no longer possible
            pos, config = nxt
            resource -= self.step_cost
            if pos in refills:
                resource, refills = self.resource_capacity, refills - {pos}
            if resource < 0:
                return None                               # the rest of the plan would run out of resource
        return c["rest"], c["reason"]

    def choose_plan(self, scene: Scene) -> Tuple[List[GameAction], str]:
        solution = self.joint_plan(scene)
        if solution:
            self.reveal_intent = None
            return solution, "execute:temporal-state-plan"
        budget = self.budget_plan(scene)
        if budget is not None:
            return budget

        experiment = self.semantic_experiment(scene)
        if experiment is not None:
            return experiment

        best = None
        for tile in self.discovery_candidates(scene):
            path = self.interaction_plan(scene, tile, 70000)
            if path is None:
                continue
            reserve = self.resource_plan(scene)
            if scene.resource < self.step_cost * (len(path) + 4) and reserve:
                return reserve, "recharge-before-discovery"
            score = (0 if tile.kind == "pickup_visual" else 1, tile.visits, len(path))
            if best is None or score < best[0]:
                best = (score, path, f"explore:tile@{tile.pos}")
        if best:
            return best[1], best[2]

        reveal = self.reveal_plan(scene)
        if reveal is not None:
            return reveal

        role = self.required_role(scene)
        for signature in self.moving_signatures:
            model = self.models.get(signature)
            track = self.moving_tracks.get(signature)
            unknown = model is None or model.role in ("unknown", "none")
            context_unknown = model is not None and self.model_matches_role(model, role or "") and model.predict(scene.key) is None
            if track is None or track.period is None or not (unknown or context_unknown):
                continue
            path = self.joint_plan(
                scene, allow_unknown_target=True, max_nodes=90000,
                mobile_target_signature=signature
            )
            if path:
                return path, f"explore:mobile@{signature}:period={track.period}"

        for source, crossing in self.unknown_gate_crossings(scene):
            path = self.plan_to_test_gate(scene, source, crossing)
            if path is not None:
                score = len(path)
                if best is None or score < best[0][2]:
                    best = ((2, 0, score), path, f"explore:dynamic@{source}:{crossing.name}")
        if best:
            return best[1], best[2]

        for pos in sorted(self.free_memory, key=lambda p: abs(p[0] - scene.player[0]) + abs(p[1] - scene.player[1])):
            semantic = (scene.level, pos, scene.key, scene.dynamic_config, self.level_tick % self.temporal_cycle())
            if self.seen[semantic] == 0:
                path = self.careful_plan(scene, pos, fallback=False)
                if path:
                    return path, "explore:frontier"
        legal = []
        for action in MOVE:
            transition = self.transition_position(scene.player, action, scene.dynamic_config)
            if transition is not None and transition[0] in self.free_memory:
                if self.careful_exploration and transition[0] in self.mechanism_cells():
                    continue
                legal.append(action)
        return ([legal[int(self.rng.integers(len(legal)))]] if legal else [GameAction.ACTION1]), "fallback"

    def next_action(self, obs, env=None) -> GameAction:
        scene = self.parse(obs)
        if scene.level != self.level:
            self.begin_level(scene)
        previous, previous_action = self.last_scene, self.last_action
        restarting = self.pending_restart
        self.merge_scene(scene)
        self.observe_transition(scene)
        if self.slide_model is not None:
            if restarting or previous_action not in MOVE:
                previous = None
            delta = MOVE.get(previous_action, (0, 0))
            fresh = self.slide_model(previous, delta, scene)
            self.slides = {c: v for c, v in self.slides.items() if c not in scene.visible_cells} | fresh
        temporal_phase = self.level_tick % self.temporal_cycle()
        semantic = (
            scene.level, scene.player, scene.key,
            scene.resource // max(1, self.step_cost), scene.dynamic_config, temporal_phase
        )
        self.seen[semantic] += 1
        if self.seen[semantic] > 1:
            self.semantic_repeats += 1
        kept = self.continue_plan(scene, restarting) if self.commit_plans else None
        if kept is not None:
            plan, self.reason = kept
            self.plan_reuses += 1
        else:
            plan, self.reason = self.choose_plan(scene)
        if self.commit_plans:
            self.commit(scene, plan, self.reason)
        self.last_plan = list(plan)
        action = plan[0] if plan else GameAction.ACTION1
        if self.verbose and self.reason != self.last_reason:
            periods = {
                sig: track.period for sig, track in self.moving_tracks.items()
                if sig in self.moving_signatures
            }
            print(
                f"  plan={self.reason} length={len(plan)} pos={scene.player} "
                f"resource={scene.resource} dynamic_edges={len(self.dynamic_edges)} "
                f"mobile_periods={periods}"
            )
        self.last_reason = self.reason
        self.last_scene = scene
        self.last_action = action
        self.actions += 1
        return action

    def export_memory(self, path: Path) -> None:
        data = {
            "tile_models": {
                signature: {
                    "role": model.role,
                    "votes": dict(model.votes),
                    "visits": model.visits,
                    "color_map": model.color_map,
                    "shape_map": {str(k): v for k, v in model.shape_map.items()},
                    "transitions": {str(k): v for k, v in model.transitions.items()},
                }
                for signature, model in self.models.items()
            },
            "dynamic_transitions": [asdict(x) for x in self.dynamic_edges.values()],
            "blocked_edges": [[list(a), list(b)] for a, b in self.blocked_edges],
            "moving_signatures": sorted(self.moving_signatures),
            "moving_tracks": {
                signature: {
                    "period": track.period,
                    "confidence": track.confidence,
                    "history": [[tick, list(pos)] for tick, pos in track.history],
                    "phase_positions": {
                        str(phase): {str(pos): count for pos, count in votes.items()}
                        for phase, votes in track.phase_positions.items()
                    },
                }
                for signature, track in self.moving_tracks.items()
                if signature in self.moving_signatures
            },
            "signature_positions": {k: [list(p) for p in sorted(v)] for k, v in self.signature_positions.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
