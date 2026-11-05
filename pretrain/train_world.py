"""Pretrain WorldNet on the game library.  python3 -m pretrain.train_world [--steps 6000]

Games are generated on the fly, so the net never sees the same one twice and cannot memorise them.
Every batch is also rotated or mirrored at random, with the arrow keys turned to match.
Writes agent/weights/world_net.pt and prints how well each head does on games it never trained on.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent.world_net import WEIGHTS, WorldNet
from pretrain.library import samples

# rotating the picture by 90 degrees anticlockwise turns up into left, left into down, ...
ROTATE = {0: 2, 2: 1, 1: 3, 3: 0}        # index of ACTION1..4 in KEYS -> index after one rotation
MIRROR = {2: 3, 3: 2}                    # left and right swap in a mirror image


def augment(batch: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for s in batch:
        k, flip = rng.randrange(4), rng.random() < 0.5
        frame, roles, clicks = s["frame"], s["roles"], s["clicks"]
        keys, available = s["keys"].copy(), s["available"].copy()
        for _ in range(k):
            frame, roles, clicks = np.rot90(frame), np.rot90(roles), np.rot90(clicks)
            keys[[ROTATE.get(i, i) for i in range(6)]] = keys.copy()
            available[[ROTATE.get(i, i) for i in range(6)]] = available.copy()
        if flip:
            frame, roles, clicks = frame[:, ::-1], roles[:, ::-1], clicks[:, ::-1]
            keys[[MIRROR.get(i, i) for i in range(6)]] = keys.copy()
            available[[MIRROR.get(i, i) for i in range(6)]] = available.copy()
        out.append({**s, "frame": frame.copy(), "roles": roles.copy(), "clicks": clicks.copy(),
                    "keys": keys, "available": available})
    return out


def batch_of(rng: random.Random, size: int) -> list[dict]:
    out = []
    while len(out) < size:
        out += samples(rng, per_game=4)
    return out[:size]


def tensors(batch: list[dict]) -> dict[str, torch.Tensor]:
    stack = lambda k, dtype: torch.from_numpy(np.stack([b[k] for b in batch]).astype(dtype))
    return {"frame": stack("frame", np.int64), "roles": stack("roles", np.int64), "clicks": stack("clicks", np.float32),
            "keys": stack("keys", np.float32), "available": stack("available", np.float32),
            "can_click": torch.tensor([b["can_click"] for b in batch]), "value": stack("value", np.float32)}


def losses(net: WorldNet, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = net(t["frame"])
    clicks = F.binary_cross_entropy_with_logits(out["clicks"], t["clicks"], reduction="none").mean((1, 2))
    keys = F.binary_cross_entropy_with_logits(out["keys"], t["keys"], reduction="none") * t["available"]
    return {"roles": F.cross_entropy(out["roles"], t["roles"]),
            "clicks": (clicks * t["can_click"]).sum() / t["can_click"].sum().clamp(min=1),
            "keys": keys.sum() / t["available"].sum().clamp(min=1),
            "value": F.binary_cross_entropy_with_logits(out["value"], t["value"])}


@torch.no_grad()
def evaluate(net: WorldNet, rng: random.Random, n: int = 512) -> dict[str, float]:
    net.eval()
    t = tensors(batch_of(rng, n))
    out = net(t["frame"])
    roles = (out["roles"].argmax(1) == t["roles"]).float()
    special = t["roles"] != 0
    click = (out["clicks"] > 0).float()
    clickable = t["can_click"] > 0
    hit = (click * t["clicks"]).sum((1, 2))[clickable]
    precision = hit / click.sum((1, 2))[clickable].clamp(min=1)
    recall = hit / t["clicks"].sum((1, 2))[clickable].clamp(min=1)
    keys_ok = (((out["keys"] > 0).float() == t["keys"]).float() * t["available"]).sum() / t["available"].sum()
    value = torch.sigmoid(out["value"])
    net.train()
    return {"role accuracy (not background)": roles[special].mean().item(),
            "click precision": precision.mean().item(), "click recall": recall.mean().item(),
            "key accuracy": keys_ok.item(), "value error": (value - t["value"]).abs().mean().item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng, test_rng = random.Random(args.seed), random.Random(10_000 + args.seed)
    net = WorldNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=args.steps)
    t0, running = time.time(), {}
    for step in range(1, args.steps + 1):
        parts = losses(net, tensors(augment(batch_of(rng, args.batch), rng)))
        loss = parts["roles"] + parts["clicks"] + parts["keys"] + 2 * parts["value"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        for k, v in parts.items():
            running[k] = 0.98 * running.get(k, v.item()) + 0.02 * v.item()
        if step % 500 == 0 or step == args.steps:
            print(f"step {step:5d}  {time.time() - t0:5.0f}s  " + "  ".join(f"{k} {v:.3f}" for k, v in running.items()),
                  flush=True)
    metrics = evaluate(net, test_rng)
    print("on new games:", "  ".join(f"{k} {v:.3f}" for k, v in metrics.items()))
    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(net.state_dict(), WEIGHTS)
    print("saved", WEIGHTS)


if __name__ == "__main__":
    main()
