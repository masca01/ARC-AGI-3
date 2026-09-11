"""Play MyAgent locally on ARC-AGI-3 games, print what it learns, optionally save a replay GIF.

    python3 run_local.py --list                     # show the public games
    python3 run_local.py --game ls20                # one game
    python3 run_local.py --game ls20,ft09 --gif     # several games + replays in outputs/
    python3 run_local.py --all --steps 200          # every public game (what Kaggle does)
    python3 run_local.py --game ls20 --render human --steps 60   # live window (slow)

Needs ARC_API_KEY in .env. Games download once and are cached in environment_files/.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import arc_agi
from arc_agi.rendering import COLOR_MAP, hex_to_rgb

from agent.my_agent import MyAgent

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
PALETTE = np.array([hex_to_rgb(COLOR_MAP[i]) for i in range(16)], dtype=np.uint8)


def fmt(v, nd=2):
    return "  -  " if v is None else f"{v:.{nd}f}"


def play(arc, game_id: str, steps: int, render, log_every: int) -> dict | None:
    env = arc.make(game_id, render_mode=render)
    if env is None:
        print(f"  could not create env for {game_id}")
        return None
    MyAgent.MAX_ACTIONS = steps
    agent = MyAgent(card_id="local", game_id=game_id, agent_name=f"MyAgent.{game_id}",
                    ROOT_URL="http://localhost", record=False, arc_env=env, tags=["local"])

    choose = agent.choose_action

    def choose_and_log(frames, latest):
        action = choose(frames, latest)
        if agent.action_counter % log_every == 0:
            s, c = agent.stats(), agent.last_choice
            print(f"  step {agent.action_counter:4d} | lvl {latest.levels_completed}/{latest.win_levels} | "
                  f"screens {s['screens']:4d} | counter px {s['static_px']:3d} | change acc {fmt(s['change_acc'])} | "
                  f"loss {fmt(s['loss'], 3)} | {c['action']:16s} p={fmt(c['p'])} | {c['why']}")
        return action

    agent.choose_action = choose_and_log  # type: ignore[method-assign]
    t0 = time.time()
    agent.main()
    final = agent.frames[-1]
    return dict(game=game_id, state=final.state.name, levels=final.levels_completed, win_levels=final.win_levels,
                actions=agent.action_counter, seconds=time.time() - t0, agent=agent, **agent.stats())


def save_gif(agent: MyAgent, game_id: str, scale: int = 6, ms: int = 120) -> None:
    """Left: the game and the chosen action. Right: what the CNN believed at that moment."""
    import matplotlib
    from PIL import Image, ImageDraw

    magma, size, cap = matplotlib.colormaps["magma"], 64 * scale, 44
    names = {1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "INTERACT", 7: "UNDO"}
    images = []
    for step, t in enumerate(agent.trace):
        left = Image.fromarray(PALETTE[t["frame"]]).resize((size, size), Image.NEAREST)
        right = Image.fromarray((magma(np.clip(t["heat"], 0, 1))[:, :, :3] * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)
        img = Image.new("RGB", (2 * size + 8, size + cap), (20, 20, 20))
        img.paste(left, (0, 0))
        img.paste(right, (size + 8, 0))
        d = ImageDraw.Draw(img)
        if t["action"] == 6:
            for ox in (0, size + 8):
                cx, cy = ox + t["x"] * scale + scale // 2, t["y"] * scale + scale // 2
                for col, wd, r in (((0, 0, 0), 4, 9), ((255, 255, 255), 2, 8)):
                    d.line([(cx - r, cy - r), (cx + r, cy + r)], fill=col, width=wd)
                    d.line([(cx - r, cy + r), (cx + r, cy - r)], fill=col, width=wd)
            label = f"CLICK ({t['x']},{t['y']})"
        else:
            label = f"ACTION{t['action']} {names.get(t['action'], '')}"
        p = "-" if t["p"] is None else f"{t['p']:.2f}"
        d.text((6, size + 4), f"step {step:3d}   level {t['level']}   {label}   P(change)={p}", fill=(255, 255, 255))
        d.text((6, size + 22), t["why"], fill=(180, 180, 180))
        d.text((size + 14, 4), t["heat_title"], fill=(255, 255, 255))
        images.append(img)
    if not images:
        return
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{game_id}_replay.gif"
    images[0].save(path, save_all=True, append_images=images[1:], duration=ms, loop=0, optimize=True)
    print(f"  saved {path.relative_to(ROOT)} ({len(images)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="ls20", help="game id, or several separated by commas")
    ap.add_argument("--all", action="store_true", help="play every public game")
    ap.add_argument("--list", action="store_true", help="list the public games and exit")
    ap.add_argument("--steps", type=int, default=300, help="max actions per game")
    ap.add_argument("--gif", action="store_true", help="save a replay GIF per game in outputs/")
    ap.add_argument("--render", default=None, choices=["human", "terminal"], help="watch live (slow)")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    logging.disable(logging.INFO)
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    if args.list:
        for e in envs:
            print(f"  {e.game_id}")
        return
    games = [e.game_id.split("-")[0] for e in envs] if args.all else [g.strip() for g in args.game.split(",")]

    results = []
    for i, g in enumerate(games, 1):
        print(f"\n=== [{i}/{len(games)}] {g} ===")
        r = play(arc, g, args.steps, args.render, args.log_every)
        if r:
            results.append(r)
            if args.gif:
                save_gif(r["agent"], g)

    print("\n======================== SUMMARY ========================")
    print(f"{'game':6s} {'levels':>7s} {'actions':>8s} {'screens':>8s} {'change acc':>11s} {'time':>7s}  {'level solved at step':22s} state")
    for r in results:
        solved = ", ".join(map(str, r["level_steps"])) or "-"
        print(f"{r['game']:6s} {r['levels']:>3d}/{r['win_levels']:<3d} {r['actions']:>8d} {r['screens']:>8d} {fmt(r['change_acc']):>11s} "
              f"{r['seconds']:>6.1f}s  {solved:22s} {r['state']}")
    total = sum(r["levels"] for r in results)
    print(f"\nlevels solved in total: {total}")


if __name__ == "__main__":
    main()
