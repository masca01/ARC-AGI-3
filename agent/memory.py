"""ObjectMemory: everything significant the agent has seen in a level, where it is, and whether it
is still usable. Game-agnostic: perception hands it detections (cell, kind, signature, payload).

Why: with limited vision (ls20 level 7) the agent sees only a disc around itself. To solve the level
it must remember the objective's pattern, where the transformer tiles are, and where every refill is.
That includes refills it already used, because the game puts them back after a lost life.

Status of a remembered item, decided only from evidence:

  present          detected there, or remembered there while its cell is dark
  hidden           not detected because the player stands on it: kept
  used             vanished right after the player stepped on it (a refill was consumed): kept,
                   but not usable until the attempt restarts
  gone             vanished from a fully lit cell without the player being involved: forgotten

`restart()` (life lost, game over) makes every used item present again. If it is not there when next
seen, it becomes gone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Cell = tuple[int, int]


@dataclass
class Item:
    cell: Cell
    kind: str                  # "goal", "tile", "refill", ...
    signature: str
    payload: Any = None        # goal pattern, learned role, ...
    status: str = "present"
    first_seen: int = 0
    last_seen: int = 0
    uses: int = 0
    history: list[tuple[int, str]] = field(default_factory=list)


class ObjectMemory:
    def __init__(self) -> None:
        self.items: dict[Cell, Item] = {}
        self.events: list[tuple[str, str]] = []

    def reset_level(self) -> None:
        self.items = {}

    def restart(self, t: int) -> None:
        """The attempt restarted (life lost / game over): used objects are expected back."""
        for item in self.items.values():
            if item.status == "used":
                self._set(t, item, "present", f"{item.kind} at {item.cell} expected back after the restart")

    def observe(self, t: int, detections: dict[Cell, tuple[str, str, Any]], visible: set[Cell],
                player: Cell, previous_player: Cell | None) -> list[tuple[str, str]]:
        """Update with this step's detections; returns events (new objects, used, gone)."""
        self.events = []
        for cell, (kind, signature, payload) in detections.items():
            item = self.items.get(cell)
            if item is None or item.signature != signature:
                if item is not None:
                    self._set(t, item, "gone", f"{item.kind} at {cell} replaced by another object")
                self.items[cell] = Item(cell, kind, signature, payload, first_seen=t, last_seen=t)
                self.events.append(("memory", f"remembered {kind} at {cell}"))
                continue
            item.last_seen, item.payload = t, payload if payload is not None else item.payload
            if item.status != "present":
                self._set(t, item, "present", f"{kind} at {cell} is back")
        for cell, item in list(self.items.items()):
            if cell in detections or item.status != "present" or cell not in visible:
                continue                                   # detected, already decided, or in the dark
            if cell == player:
                continue                                   # hidden under the player
            if cell == previous_player:
                item.uses += 1
                self._set(t, item, "used", f"{item.kind} at {cell} used (vanished after the player touched it)")
            else:
                self._set(t, item, "gone", f"{item.kind} at {cell} gone")
                del self.items[cell]
        return self.events

    def mark_used(self, t: int, cell: Cell) -> None:
        item = self.items.get(cell)
        if item is not None and item.status == "present":
            item.uses += 1
            self._set(t, item, "used", f"{item.kind} at {cell} used")

    def _set(self, t: int, item: Item, status: str, text: str) -> None:
        item.status = status
        item.history.append((t, status))
        self.events.append(("memory", text))

    # ------------------------------------------------------------------ queries
    def of_kind(self, kind: str, status: str | None = None) -> list[Item]:
        return [i for i in self.items.values() if i.kind == kind and (status is None or i.status == status)]

    def summary(self) -> str:
        parts = []
        for kind in sorted({i.kind for i in self.items.values()}):
            items = self.of_kind(kind)
            usable = sum(i.status == "present" for i in items)
            parts.append(f"{kind}: {usable}/{len(items)} usable")
        return ", ".join(parts) or "-"
