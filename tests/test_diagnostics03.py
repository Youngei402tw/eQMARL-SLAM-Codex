import numpy as np
import pytest

from eqmarl_slam.diagnostics03 import covariance_summary, fixed_frame_metrics, fixed_landmark_metrics


def test_trajectory_error_uses_history_and_fixed_frame():
    truth = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    estimate = truth.copy()
    estimate[1, 1] = 3
    result = fixed_frame_metrics(truth, estimate)
    assert result["current_translation_error_m"] == 0
    assert result["translation_ate_rmse_m"] == pytest.approx(np.sqrt(3))
    shifted = fixed_frame_metrics(truth, truth + [2, 0, 0])
    assert shifted["translation_ate_rmse_m"] == 2
    assert shifted["translation_rpe_rmse_m"] == 0


def test_unobserved_landmarks_are_not_zero_error():
    result = fixed_landmark_metrics({0: [0, 0], 1: [1, 1]}, {0: [0, 0]})
    assert result["landmark_recall"] == .5
    assert result["observed_landmark_rmse_m"] == 0
    assert result["fixed_domain_landmark_rmse_m"] is None
    assert result["missing_landmark_ids"] == [1]


def test_covariance_requires_finite_correct_dimension():
    with pytest.raises(ValueError):
        covariance_summary(np.eye(2))
    assert not covariance_summary(np.diag([1., -1., 1.]))["positive_definite"]
