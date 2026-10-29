import torch

from eqmarl_slam.nonlocal_training import METHODS, expected_win, initialize, restore, train


def test_every_method_trains_and_round_trips():
    for method in METHODS:
        before = expected_win(method, initialize(method, 7))
        result = train(method, 7, 0.1, 50)
        after = expected_win(method, restore(method, result["parameters"]))
        assert float(after.detach()) >= float(before.detach()) - 1e-12
        assert abs(float(after) - result["final_expected_win"]) <= 1e-15
        assert result["gradients_finite"]


def test_entanglement_is_the_only_quantum_policy_intervention():
    separable = initialize("quantum_separable", 99)
    entangled = initialize("quantum_entangled", 99)
    assert all(torch.equal(separable[name], entangled[name]) for name in separable)
    assert expected_win("quantum_separable", separable) != expected_win("quantum_entangled", entangled)
