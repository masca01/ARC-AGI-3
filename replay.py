"""Interactive HTML replay of one game: scrub through every step and see what the agent saw,
believed and planned. Written by run_local.py --html; open the file in any browser (works offline).

    from replay import save_html
    save_html(agent, "ls20", "outputs/ls20_causal-neural.html", summary={...})

Per step it shows the game frame with overlays (planned path, target, learned tiles, where the CNN
and the hand-made detector put the player, the last motion label), every CNN output map with the
probability under the mouse, the CNN's layer activations, the action probabilities (MyAgent), the
reason for the move, and a timeline of events (levels solved, game overs, rules learned, CNN misses).
"""
from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np

TEMPLATE = Path(__file__).resolve().parent / "replay_template.html"


def _pack(array: np.ndarray) -> str:
    """uint8 array -> zlib -> base64 (the page inflates it with DecompressionStream)."""
    return base64.b64encode(zlib.compress(np.ascontiguousarray(array, dtype=np.uint8).tobytes(), 6)).decode()


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if v is None or isinstance(v, (int, str)):
        return v
    return str(v)


def _action_name(t: dict) -> str:
    if "action_name" in t:
        name = t["action_name"]
    else:
        name = "RESET" if t["action"] == 0 else f"ACTION{t['action']}"
    return f"{name} @({t['x']},{t['y']})" if t.get("x", -1) >= 0 else name


def save_html(agent, game_id: str, path: Path | str, summary: dict | None = None) -> Path:
    trace = agent.trace
    if not trace:
        raise ValueError("the agent recorded no steps")
    T = len(trace)
    n_maps = max(len(t.get("maps") or []) for t in trace) or 1
    n_layers = max(len(t.get("layers") or []) for t in trace)
    layer_size = next((t["layers"][0][1].shape[0] for t in trace if t.get("layers")), 32)

    frames = np.stack([t["frame"] for t in trace]).astype(np.uint8)
    maps = np.zeros((T, n_maps, 64, 64), np.uint8)
    layers = np.zeros((T, max(1, n_layers), layer_size, layer_size), np.uint8)
    motion = np.zeros((T, 64, 64), np.uint8)
    meta = []
    for i, t in enumerate(trace):
        step_maps = t.get("maps")
        if step_maps is None and t.get("heat") is not None and np.any(t["heat"]):
            step_maps = [(t.get("heat_title", "CNN"), t["heat"])]
        for j, (_, m) in enumerate(step_maps or []):
            maps[i, j] = np.clip(np.asarray(m, np.float32) * 255, 0, 255)
        for j, (_, m) in enumerate(t.get("layers") or []):
            layers[i, j] = np.clip(np.asarray(m, np.float32) * 255, 0, 255)
        overlay = dict(t.get("overlay") or {})
        if overlay.get("motion") is not None:
            motion[i] = overlay.pop("motion")
            overlay["motion"] = True
        if t.get("x", -1) >= 0:
            overlay["click"] = (t["y"], t["x"])
        meta.append(_jsonable(dict(
            action=_action_name(t), why=t.get("why", ""), level=t.get("level", 0), attempt=t.get("attempt", 1),
            events=t.get("events", []), overlay=overlay, info=t.get("info", {}), stats=t.get("stats", {}),
            key=t.get("key"), goals=t.get("goals"), probs=t.get("probs"),
            maps=[title for title, _ in (step_maps or [])], layers=[title for title, _ in (t.get("layers") or [])],
        )))

    from arc_agi.rendering import COLOR_MAP
    data = dict(
        game=game_id, agent=type(agent).__name__, steps=T, n_maps=n_maps, n_layers=n_layers,
        layer_size=layer_size, palette=[COLOR_MAP[i] for i in range(16)], summary=_jsonable(summary or {}),
        meta=meta, frames=_pack(frames), maps=_pack(maps), layers=_pack(layers), motion=_pack(motion),
    )
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
