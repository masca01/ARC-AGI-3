"""Regression tests for the ls20 causal agent.

    python3 tests/test_ls20_regression.py          # or: python3 -m pytest tests

The heuristic port must reproduce the teammate's original run exactly (levels 1-4 in
13 / 45 / 41 / 148 actions), so a change meant for level 5 cannot silently break levels 1-4.
Needs the cached game in environment_files/ (run run_local.py once to download it).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.causal_agent import CausalAgent  # noqa: E402
from agent.control_net import find_motion  # noqa: E402
from agent.ls20_specialist import BG, LS20ExtendedStateAgent  # noqa: E402

ORIGINAL_LEVEL_ACTIONS = [13, 45, 41, 148]


def play_ls20(perception: str, max_actions: int, oracle: bool = False) -> CausalAgent:
    import arc_agi
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    env = arc_agi.Arcade().make("ls20", include_frame_data=True)
    agent_cls = type("TestAgent", (CausalAgent,), {"PERCEPTION": perception, "ORACLE": oracle,
                                                  "MAX_ACTIONS": max_actions})
    agent = agent_cls(card_id="local", game_id="ls20", agent_name="test", ROOT_URL="http://localhost",
                      record=False, arc_env=env)
    agent.main()
    return agent


def test_heuristic_port_reproduces_original_run():
    agent = play_ls20("heuristic", max_actions=460)
    assert agent.level_actions[:4] == ORIGINAL_LEVEL_ACTIONS, agent.level_actions


def test_neural_perception_matches_hand_made_detector():
    """The CNN replaces the player detector: same levels, only the warm-up moves are extra."""
    agent = play_ls20("neural", max_actions=460, oracle=True)
    stats = agent.stats()
    assert agent.frames[-1].levels_completed >= 4, stats["level_actions"]
    assert stats["level_actions"][1:4] == ORIGINAL_LEVEL_ACTIONS[1:4], stats["level_actions"]
    assert stats["level_actions"][0] == ORIGINAL_LEVEL_ACTIONS[0] + stats["warmup_moves"]
    assert stats["player_acc"] == 1.0, stats["player_acc_by_level"]


def test_slide_net_learns_pushes_and_solves_level_5():
    """SlideNet: levels 1-3 as before, level 5 solved, most pushes predicted before they happen."""
    import arc_agi
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    env = arc_agi.Arcade().make("ls20", include_frame_data=True)
    agent_cls = type("TestAgent", (CausalAgent,), {"PERCEPTION": "neural", "SLIDES": True, "MAX_ACTIONS": 340})
    agent = agent_cls(card_id="local", game_id="ls20", agent_name="test", ROOT_URL="http://localhost",
                      record=False, arc_env=env)
    agent.main()
    slides = agent.stats()["slides"]
    assert agent.frames[-1].levels_completed >= 5, agent.level_actions
    assert agent.level_actions[1:3] == ORIGINAL_LEVEL_ACTIONS[1:3], agent.level_actions
    assert slides["false alarms"] == 0, slides
    assert slides["pushes predicted before they happened"] >= 0.75 * slides["pushes seen"], slides


def test_slide_augmentation_rotates_direction_with_the_icon():
    from agent.slide_net import NAMES, SIZE, augment

    x = np.zeros((17, SIZE, SIZE), np.float32)
    x[1, 7, 0] = 1                                   # icon on the left edge pushes right
    away = {"left": "right", "right": "left", "up": "down", "down": "up"}
    for img, label in augment(x, NAMES.index("right")):
        y, xx = map(int, np.argwhere(img[1])[0])
        side = "left" if xx == 0 else "right" if xx == SIZE - 1 else "up" if y == 0 else "down"
        assert NAMES[label] == away[side], (side, NAMES[label])


def test_rotation_classification_unchanged():
    key = (9, 9, 9, 9, BG, BG, 9, BG, 9)
    rotated = tuple(map(int, np.rot90(np.asarray(key).reshape(3, 3), -1).ravel()))
    assert LS20ExtendedStateAgent.classify_key_transition(key, rotated) == "rotate_cw"


def test_motion_label_finds_moving_block_not_floor():
    before = np.full((64, 64), 3, dtype=np.int8)
    before[20:25, 30:35] = 12                      # a 5x5 object on a uniform floor
    after = before.copy()
    after[20:25, 30:35] = 3
    after[15:20, 30:35] = 12                       # moved up by 5
    after[61, 40] = 11                             # plus a counter pixel
    m = find_motion(before, after)
    assert m is not None and m.shift == (-5, 0)
    assert m.moved[20:25, 30:35].all() and m.moved.sum() == 25
    assert m.by_itself[61, 40] and m.by_itself.sum() == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"passed  {name}")
