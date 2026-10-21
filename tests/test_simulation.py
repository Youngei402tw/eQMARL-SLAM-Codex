import numpy as np
import pytest

from eqmarl_slam.simulation import LandmarkSimulator, between_poses, compose_pose, coverage_metrics, generate_landmarks, landmark_error_metrics, scripted_commands, trajectory_error_metrics


def test_body_frame_motion_and_sensor_measurements_do_not_leak_world_truth():
    simulator = LandmarkSimulator([[1., 2., np.pi / 2]], {7: [1., 4.]}, motion_std=[0, 0, 0], odom_std=[0, 0, 0], sensor_std=[0, 0], prior_means=[[.9, 2.1, 1.5]])
    measurement = simulator.observe()[0]
    np.testing.assert_allclose(measurement["value"], [2., 0.], atol=1e-12)
    assert "landmark_position" not in measurement and "truth" not in measurement
    assert measurement["landmark_id"] == 7
    assert simulator.priors[0]["value"] == [.9, 2.1, 1.5]
    factors = simulator.step([[1., 0., 0.]])
    odometry = factors[0]
    np.testing.assert_allclose(odometry["value"], [1., 0., 0.], atol=1e-12)
    np.testing.assert_allclose(simulator.truth_poses()[0], [1., 3., np.pi / 2], atol=1e-12)
    assert odometry["from_pose"] == [0, 0] and odometry["to_pose"] == [0, 1]
    assert odometry["units"] == ["m_body", "m_body", "rad"]


def test_keyed_noise_is_call_order_independent_and_streams_are_distinct():
    kwargs = {"initial_poses": [[0, 0, 0]], "landmarks": {0: [1, 1], 1: [2, 2]}, "seeds": {"motion": 51, "odometry": 52, "sensor": 53}}
    first, second = LandmarkSimulator(**kwargs), LandmarkSimulator(**kwargs)
    first.observe()
    first.observe()
    a = first.step([[.5, 0., .1]])
    b = second.step([[.5, 0., .1]])
    assert a == b
    assert first.observe() == first.observe()
    assert len({measurement["measurement_id"] for measurement in a}) == len(a)
    np.testing.assert_array_equal(first.truth_trajectory(0), second.truth_trajectory(0))
    changed = LandmarkSimulator(**{**kwargs, "seeds": {"motion": 51, "odometry": 52, "sensor": 54}})
    c = changed.step([[.5, 0., .1]])
    assert c[0] == a[0] and c[1:] != a[1:]
    np.testing.assert_array_equal(first.truth_poses(), changed.truth_poses())


def test_closed_loop_and_revisit_scripts_close_without_noise():
    for kind in ["closed_loop", "revisit"]:
        pose = np.array([0., 0., 0.])
        for command in scripted_commands(kind):
            pose = compose_pose(pose, command)
        np.testing.assert_allclose(pose, [0., 0., 0.], atol=1e-12)
    start = np.array([2., -1., .7])
    delta = np.array([.8, -.3, .4])
    np.testing.assert_allclose(between_poses(start, compose_pose(start, delta)), delta, atol=1e-12)


def test_relative_pose_noise_model_and_range_are_explicit():
    simulator = LandmarkSimulator([[0, 0, np.pi / 2], [1, 2, .3]], {0: [3, 3]}, relative_std=[0, 0, 0])
    measurement = simulator.relative_measurement(0, 1)
    np.testing.assert_allclose(measurement["value"], [2., -1., .3 - np.pi / 2], atol=1e-12)
    assert measurement["from_pose"] == [0, 0] and measurement["to_pose"] == [1, 0]
    assert simulator.relative_measurement(0, 1, max_range=1.) is None
    with pytest.raises(ValueError, match="distinct"):
        simulator.relative_measurement(0, 0)


def test_fixed_frame_trajectory_and_fixed_id_landmark_metrics_retain_missing():
    truth = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    estimate = truth.copy()
    estimate[:, :2] += [3, 4]
    metrics = trajectory_error_metrics(estimate, truth)
    assert metrics["trajectory_ate_translation_m"] == 5
    assert metrics["current_pose_translation_error_m"] == 5
    assert metrics["trajectory_rpe_translation_m"] == 0
    assert metrics["frame_convention"] == "fixed_declared_world_frame_no_alignment"
    landmarks = {0: [0, 0], 1: [1, 0]}
    missing = landmark_error_metrics({0: [0, 0]}, landmarks, [0, 1])
    assert missing["landmark_rmse_fixed_ids_m"] is None
    assert missing["landmark_estimate_coverage"] == .5
    penalized = landmark_error_metrics({0: [0, 0]}, landmarks, [0, 1], missing_penalty_m=2.)
    assert penalized["landmark_rmse_fixed_ids_m"] == pytest.approx(np.sqrt(2))


def test_physical_coverage_is_independent_of_estimated_pose_drift():
    truth = [np.array([[.5, .5, 0.]])]
    near = [np.array([[.5, .5, 0.], [1.5, .5, 0.]])]
    metrics = coverage_metrics(truth, near, bounds=[0, 2, 0, 1], resolution=1., sensor_range=.6)
    assert metrics["fixed_domain_cells"] == 2
    assert metrics["physical_coverage"] == .5
    assert metrics["estimated_coverage"] == 1.


def test_map_and_prior_sampling_are_reproducible_with_separate_streams():
    first, second = generate_landmarks(123), generate_landmarks(123)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    simulator = LandmarkSimulator([[0, 0, 0]], first, sample_initial_prior=True)
    assert not np.array_equal(simulator.truth_poses()[0], simulator.priors[0]["value"])
    assert simulator.priors[0]["covariance"] == pytest.approx([.05 ** 2, .05 ** 2, .02 ** 2])
