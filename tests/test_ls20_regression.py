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
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arcengine import GameState  # noqa: E402

from agent.causal_agent import CausalAgent  # noqa: E402
from agent.control_net import find_motion  # noqa: E402
from agent.general_agent import GeneralAgent  # noqa: E402
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
    assert slides["false alarms"] <= 2, slides          # a learning model may be wrong once; it fast-learns
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


def test_objectives_move_on_to_the_second_door():
    """Level 6 has two doors: after the first one disappears the agent must go for the second."""
    import arc_agi
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    env = arc_agi.Arcade().make("ls20", include_frame_data=True)
    agent_cls = type("TestAgent", (CausalAgent,), {"PERCEPTION": "neural", "SLIDES": True, "MAX_ACTIONS": 430})
    agent = agent_cls(card_id="local", game_id="ls20", agent_name="test", ROOT_URL="http://localhost",
                      record=False, arc_env=env)
    agent.main()
    assert agent.frames[-1].levels_completed >= 6, agent.level_actions
    achieved = [t for step in agent.trace if step["level"] == 5 for k, t in step["events"] if k == "goal"]
    assert any("achieved" in t for t in achieved), achieved


def test_full_game_won_with_limited_vision_level_7():
    """Level 7 shows only a disc around the player: explore, remember the objective, tiles, refills and
    the moving rotator's cycle, and win the whole game."""
    import arc_agi
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    env = arc_agi.Arcade().make("ls20", include_frame_data=True)
    agent_cls = type("TestAgent", (CausalAgent,), {"PERCEPTION": "neural", "SLIDES": True, "MAX_ACTIONS": 700})
    agent = agent_cls(card_id="local", game_id="ls20", agent_name="test", ROOT_URL="http://localhost",
                      record=False, arc_env=env)
    agent.main()
    assert agent.frames[-1].state.name == "WIN", agent.level_actions
    assert agent.frames[-1].levels_completed == 7
    # plan commitment: no stepping back and forth (443 actions; 611 when replanning every step)
    assert sum(agent.level_actions[:7]) <= 480, agent.level_actions


def test_learned_gauge_wins_the_game_with_no_fixed_hud_reading():
    """The energy bar is discovered, not read at a fixed place: same win, every reading correct."""
    import arc_agi
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    env = arc_agi.Arcade().make("ls20", include_frame_data=True)
    agent_cls = type("TestAgent", (CausalAgent,), {"PERCEPTION": "neural", "SLIDES": True, "ORACLE": True,
                                                  "GAUGE": True, "MAX_ACTIONS": 700})
    agent = agent_cls(card_id="local", game_id="ls20", agent_name="test", ROOT_URL="http://localhost",
                      record=False, arc_env=env)
    agent.main()
    gauge = agent.stats()["gauge"]
    assert agent.frames[-1].state.name == "WIN", agent.level_actions
    assert sum(agent.level_actions[:7]) <= 480, agent.level_actions
    assert gauge["box"] == (61, 13, 62, 54) and gauge["capacity"] == 42, gauge
    assert gauge["reads correct"] == gauge["reads checked"] > 400, gauge


def test_gauge_finder_needs_no_place_or_colour():
    """A vertical bar at another place and colour is found; a moving object is not a gauge."""
    from agent.gauge import GaugeFinder

    frame = np.full((64, 64), 3, dtype=np.int8)
    frame[10:40, 2:4] = 7                                   # a vertical bar, draining from the top
    finder = GaugeFinder()
    for i in range(5):
        after = frame.copy()
        after[10 + i, 2:4] = 3
        finder.observe(frame, after)
        frame = after
    assert finder.gauge is not None and finder.gauge.rect == (10, 2, 39, 3), finder.gauge
    assert finder.capacity == 30 and finder.read(frame) == 25

    frame = np.full((64, 64), 3, dtype=np.int8)
    frame[30:32, 10:20] = 7                                 # a block sliding along: its colour never shrinks
    finder = GaugeFinder()
    for i in range(8):
        after = np.roll(frame, 1, axis=1)
        finder.observe(frame, after)
        frame = after
    assert finder.gauge is None


def test_gauge_learns_what_empty_means_from_a_restart():
    from agent.gauge import GaugeFinder

    frame = np.full((64, 64), 3, dtype=np.int8)
    frame[60, 10:30] = 7
    finder = GaugeFinder()
    assert finder.fatal_when_empty is None
    for i in range(4):
        after = frame.copy()
        after[60, 10 + i] = 3
        finder.observe(frame, after)
        frame = after
    assert finder.observe_restart(0) and finder.fatal_when_empty is True


def play_general(game: str, max_actions: int):
    import arc_agi
    from dotenv import load_dotenv


    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    env = arc_agi.Arcade().make(game, include_frame_data=True)
    agent_cls = type("TestAgent", (GeneralAgent,), {"MAX_ACTIONS": max_actions})
    agent = agent_cls(card_id="local", game_id=game, agent_name="test", ROOT_URL="http://localhost",
                      record=False, arc_env=env)
    agent.main()
    return agent


def test_general_agent_solves_a_first_level_of_keyboard_and_click_games():
    """One agent with no game knowledge: a keyboard game and a click game, seed 0."""
    assert play_general("ls20", 400).frames[-1].levels_completed >= 1
    assert play_general("r11l", 60).frames[-1].levels_completed >= 1


def test_general_agent_objects_leave_out_background_and_ignored_pixels():
    from agent.general_agent import objects

    frame = np.full((64, 64), 3, dtype=np.uint8)
    frame[10:14, 10:14] = 7
    frame[50:54, 50:54] = 7
    frame[63, :] = 9
    ignore = np.zeros((64, 64), bool)
    ignore[63, :] = True
    targets = objects(frame, ignore)
    assert len(targets) == 2                                # no background, no ignored bar
    assert targets[0].kind == targets[1].kind               # same look, same kind of object
    assert all(frame[y, x] == 7 for x, y in (t.action for t in targets))


def test_general_agent_routes_must_be_affordable():
    from agent.general_agent import LevelGraph, Node

    blank = np.zeros((64, 64), dtype=np.uint8)
    graph = LevelGraph()
    graph.nodes = {k: Node([], blank) for k in (b"a", b"b", b"c", b"refill")}
    graph.nodes[b"a"].edges = {1: (b"b", 5), 2: (b"refill", 1)}
    graph.nodes[b"refill"].edges = {3: (b"b", -20)}
    graph.nodes[b"b"].edges = {4: (b"c", 8)}
    goal = lambda key: key == b"c"
    assert graph.route(b"a", goal, 20) == [1, 4]            # shortest, when it is affordable
    assert graph.route(b"a", goal, 10) == [2, 3, 4]         # through the refill, when it is not
    assert graph.route(b"a", goal, 1) is None


def test_click_game_falls_back_to_the_general_explorer():
    """A game without arrow keys (ft09) must not crash: the agent hands over to GeneralAgent."""
    import arc_agi
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    logging.disable(logging.CRITICAL)
    env = arc_agi.Arcade().make("ft09", include_frame_data=True)
    agent_cls = type("TestAgent", (CausalAgent,), {"PERCEPTION": "neural", "SLIDES": True, "MAX_ACTIONS": 30})
    agent = agent_cls(card_id="local", game_id="ft09", agent_name="test", ROOT_URL="http://localhost",
                      record=False, arc_env=env)
    agent.main()
    assert agent.fallback == "the game has no arrow keys"
    assert isinstance(agent.explorer, GeneralAgent)
    assert agent.action_counter > 30


def test_general_agent_taking_over_midway_does_not_take_that_frame_for_the_start():
    """Taken over mid-level, the first frame is not a start, or the player's cells get ignored at the next restart."""
    agent = GeneralAgent(card_id="local", game_id="test", agent_name="test", ROOT_URL="http://localhost",
                         record=False, arc_env=None)
    agent.take_over(0, midway=True)
    playing = SimpleNamespace(state=GameState.NOT_FINISHED, levels_completed=0, available_actions=[1, 2],
                              frame=[np.zeros((64, 64), int).tolist()])
    agent.choose_action([], playing)
    assert agent.level_start is None
    agent.choose_action([], SimpleNamespace(**{**vars(playing), "state": GameState.GAME_OVER}))
    start = np.zeros((64, 64), int)
    start[10, 10] = 3
    agent.choose_action([], SimpleNamespace(**{**vars(playing), "frame": [start.tolist()]}))
    assert (agent.level_start == start).all() and not agent.ignore.any()


def test_object_memory_keeps_used_refills_for_the_next_attempt():
    from agent.memory import ObjectMemory

    memory = ObjectMemory()
    refill = {(2, 2): ("refill", "r", None)}
    memory.observe(0, refill, {(2, 2), (2, 3)}, (2, 3), None)
    memory.observe(1, {}, {(2, 2), (2, 3)}, (2, 2), (2, 3))       # player on it: hidden, kept
    assert memory.items[(2, 2)].status == "present"
    memory.observe(2, {}, {(2, 2), (2, 3)}, (2, 3), (2, 2))       # vanished after the touch: used
    assert memory.items[(2, 2)].status == "used"
    memory.observe(3, {}, set(), (5, 5), (2, 3))                  # in the dark: still remembered
    memory.restart(4)                                             # lost a life: it comes back
    assert memory.items[(2, 2)].status == "present"
    memory.observe(5, {}, {(2, 2)}, (5, 5), (5, 4))               # seen empty, player not involved: gone
    assert (2, 2) not in memory.items


def test_objective_tracker_hidden_gone_reopened_no_effect():
    from agent.objectives import ObjectiveTracker

    frame = np.full((64, 64), 3, dtype=np.int8)
    frame[9:16, 9:16] = 5                                   # a 7x7 goal box
    box, tracker = (9, 9, 15, 15), ObjectiveTracker()
    tracker.update(0, {"door": ((2, 2), box, None)}, (0, 0), frame, None)
    covered = frame.copy()
    covered[10:15, 10:15] = 12                              # the player stands on it: box not detected,
    tracker.update(1, {}, (2, 2), covered, (10, 10, 14, 14))  # but its border is unchanged -> hidden
    assert tracker.is_open("door")
    gone = covered.copy()
    gone[9:16, 9] = gone[9:16, 15] = gone[9, 9:16] = gone[15, 9:16] = 3   # border replaced by floor
    tracker.update(2, {}, (2, 2), gone, (10, 10, 14, 14))
    assert not tracker.is_open("door")                      # gone, even with the player on top of it
    tracker.update(3, {"door": ((2, 2), box, None)}, (0, 0), frame, None)
    assert tracker.is_open("door")                          # back after a life was lost: open again
    for t in range(4, 8):                                   # stand on it twice without effect
        player = (2, 2) if t % 2 == 0 else (0, 0)
        tracker.update(t, {"door": ((2, 2), box, None)}, player, frame, None)
    assert tracker.penalty("door") == 1


def test_darkness_is_border_background_not_enclosed_boxes():
    frame = np.full((64, 64), BG, dtype=np.int8)            # limited vision: everything dark...
    frame[10:40, 10:40] = 3                                 # ...except a lit area around the player
    frame[20:27, 20:27] = BG                                # an enclosed background-coloured box (a door)
    dark = LS20ExtendedStateAgent.darkness(frame)
    assert dark[0, 0] and dark[50, 50] and not dark[15, 15]
    assert not dark[23, 23]                                 # the enclosed box is visible, not darkness


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
