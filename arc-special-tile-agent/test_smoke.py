import numpy as np

from agent import BG, LS20ExtendedStateAgent, TileModel


def test_rotation_classification():
    key = (9, 9, 9, 9, BG, BG, 9, BG, 9)
    rotated = tuple(map(int, np.rot90(np.asarray(key).reshape(3, 3), -1).ravel()))
    assert LS20ExtendedStateAgent.classify_key_transition(key, rotated) == "rotate_cw"


def test_recolor_classification():
    before = (9, 9, BG, BG, 9, BG, 9, BG, BG)
    after = tuple(12 if value == 9 else value for value in before)
    assert LS20ExtendedStateAgent.classify_key_transition(before, after) == "recolor"


def test_learned_rotation_prediction():
    key = (9, 9, 9, 9, BG, BG, 9, BG, 9)
    model = TileModel("rotation-tile")
    model.votes["rotate_cw"] = 1
    expected = tuple(map(int, np.rot90(np.asarray(key).reshape(3, 3), -1).ravel()))
    assert model.predict(key) == expected


def test_episode_state_initialization():
    agent = LS20ExtendedStateAgent(max_actions=250, seed=7, verbose=False)
    assert agent.max_actions == 250
    assert agent.level == -1
    assert agent.actions == 0
    assert agent.dynamic_edges == {}
    assert agent.reason == "initializing"


if __name__ == "__main__":
    test_rotation_classification()
    test_recolor_classification()
    test_learned_rotation_prediction()
    test_episode_state_initialization()
    print("All smoke tests passed.")
