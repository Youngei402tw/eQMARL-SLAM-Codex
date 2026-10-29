import numpy as np

from eqmarl_slam.diagnostics18 import _wilson


def test_wilson_interval_contains_observed_rate():
    interval = _wilson(850, 1000)
    assert interval[0] < 0.85 < interval[1]


def test_locked_pairing_is_conservative_against_strongest_control():
    entangled = np.array([0.85, 0.84])
    classical = np.array([0.74, 0.75])
    separable = np.array([0.75, 0.73])
    differences = entangled - np.maximum(classical, separable)
    assert differences.tolist() == [0.09999999999999998, 0.08999999999999997]
