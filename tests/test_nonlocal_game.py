import math

import torch

from eqmarl_slam.nonlocal_game import (
    CLASSICAL_CEILING, TSIRELSON_WIN, bell_optimal_angles,
    classical_local_expected_win, deterministic_strategy_scores,
    quantum_expected_win, reward, shared_randomness_expected_win,
)


def test_reward_and_exhaustive_classical_ceiling():
    assert reward(1, 1, 0, 1) == 1
    assert reward(1, 1, 0, 0) == 0
    scores = deterministic_strategy_scores()
    assert len(scores) == 16
    assert max(row["expected_win"] for row in scores) == CLASSICAL_CEILING


def test_bell_policy_reaches_tsirelson_win_and_separable_control_does_not():
    alpha, beta = bell_optimal_angles()
    assert abs(float(quantum_expected_win(alpha, beta, entangled=True)) - TSIRELSON_WIN) < 1e-12
    zero = torch.zeros(2, dtype=torch.float64)
    assert float(quantum_expected_win(zero, zero, entangled=False)) == CLASSICAL_CEILING


def test_classical_policy_classes_obey_information_bound():
    local = classical_local_expected_win(torch.tensor([-30.0, -30.0]), torch.tensor([-30.0, -30.0]))
    assert abs(float(local) - CLASSICAL_CEILING) < 1e-12
    logits = torch.full((16,), -30.0, dtype=torch.float64)
    best = max(range(16), key=lambda index: deterministic_strategy_scores()[index]["expected_win"])
    logits[best] = 30.0
    assert float(shared_randomness_expected_win(logits)) <= CLASSICAL_CEILING


def test_noise_reduces_bell_advantage_monotonically():
    alpha, beta = bell_optimal_angles()
    values = [float(quantum_expected_win(alpha, beta, entangled=True, noise=p)) for p in (0.0, 0.02, 0.05)]
    assert values[0] > values[1] > values[2] > 0.5
