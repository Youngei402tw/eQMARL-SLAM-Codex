"""Meaningful estimator contracts, independent of reportable scripted runs."""
import importlib.util
import sys

import numpy as np
import pytest

from eqmarl_slam.slam import GTSAMSmoother, MissingPoseError


pytestmark = pytest.mark.skipif(importlib.util.find_spec("gtsam") is None,
                                reason="optional GTSAM backend is not installed")


def initialized(**kwargs):
    smoother = GTSAMSmoother(backend="direct", **kwargs)
    smoother.add_prior(0, [0, 0, 0], "prior:0")
    return smoother


def test_unknown_landmark_initialization_has_no_true_coordinate_prior():
    smoother = initialized()
    before = smoother.pose_covariance(0, 0)
    smoother.add_observation(0, 0, 7, np.pi / 4, np.sqrt(8), "obs:0:7")
    np.testing.assert_allclose(smoother.landmarks()[7], [2, 2], atol=1e-9)
    np.testing.assert_allclose(smoother.pose_covariance(0, 0), before, atol=1e-9)
    assert np.linalg.eigvalsh(smoother.landmark_covariance(7)).min() > 0


def test_body_frame_motion_and_world_covariance_convention():
    smoother = GTSAMSmoother(prior_sigmas=[.1, .2, .05], backend="direct")
    smoother.add_prior(0, [2, 3, np.pi / 2], "prior")
    smoother.add_odometry(0, 1, [1, 0, 0], "odom")
    np.testing.assert_allclose(smoother.pose(0, 1), [2, 4, np.pi / 2], atol=1e-9)
    local = smoother.pose_covariance(0, 0, world_frame=False)
    world = smoother.pose_covariance(0, 0, world_frame=True)
    np.testing.assert_allclose(np.diag(local), [.01, .04, .0025], atol=1e-10)
    np.testing.assert_allclose(np.diag(world), [.04, .01, .0025], atol=1e-10)


def test_known_landmark_revisit_reduces_same_step_pose_uncertainty():
    smoother = initialized(odometry_sigmas=[.2, .2, .1])
    smoother.add_observation(0, 0, 0, .4, 3, "initial_observation")
    smoother.add_odometry(0, 1, [1, 0, 0], "odom")
    prediction = smoother.clone()
    smoother.add_observation(0, 1, 0, .6, 2.2, "revisit")
    assert np.trace(smoother.pose_covariance(0, 1)) < np.trace(prediction.pose_covariance(0, 1))
    assert set(smoother.trajectory(0)) == {0, 1}


def test_duplicate_and_conflicting_factors_cannot_invent_evidence():
    smoother = initialized()
    smoother.add_observation(0, 0, 0, .5, 2, "obs")
    before = smoother.pose_covariance(0, 0)
    count = smoother.factor_count
    assert not smoother.add_observation(0, 0, 0, .5, 2, "obs")
    assert smoother.factor_count == count
    np.testing.assert_array_equal(smoother.pose_covariance(0, 0), before)
    with pytest.raises(ValueError, match="Conflicting"):
        smoother.add_observation(0, 0, 0, .5, 2.1, "obs")
    assert smoother.factor_count == count


def test_raw_factor_delivery_unions_once_and_handles_dependency_order():
    sender = initialized()
    sender.add_odometry(0, 1, [1, 0, .1], "odom")
    sender.add_observation(0, 1, 12, .2, 3, "obs")
    receiver = GTSAMSmoother(backend="direct")
    assert receiver.merge_factors(reversed(sender.export_factors())) == 3
    assert receiver.merge_factors(sender.export_factors()) == 0
    np.testing.assert_allclose(sender.pose(0, 1), receiver.pose(0, 1), atol=1e-9)
    np.testing.assert_allclose(sender.pose_covariance(0, 1), receiver.pose_covariance(0, 1), atol=1e-9)
    with pytest.raises(MissingPoseError):
        GTSAMSmoother(backend="direct").merge_factors([sender.export_factors()[1]])


def test_relative_robot_factor_preserves_joint_correlations():
    smoother = initialized()
    smoother.add_prior(1, [2, 0, 0], "prior:1", sigmas=[1, 1, .5])
    before = smoother.pose_covariance(1, 0)
    smoother.add_relative(0, 0, 1, 0, [2, 0, 0], "relative")
    assert np.trace(smoother.pose_covariance(1, 0)) < np.trace(before)
    assert smoother.factor_count == 3


def test_loop_measurement_can_revise_historical_pose():
    smoother = initialized(odometry_sigmas=[.1, .1, .05])
    smoother.add_observation(0, 0, 0, 0, 3, "obs:initial")
    for step in range(1, 5):
        # An equal-length square closes even when all four edges have a
        # common scale bias. Perturb one edge to make the closure observable.
        smoother.add_odometry(0, step, [1.2 if step == 2 else 1.0, 0, np.pi / 2], f"odom:{step}")
    historical_before = smoother.pose(0, 2).copy()
    smoother.add_observation(0, 4, 0, 0, 3, "obs:closure")
    historical_after = smoother.pose(0, 2)
    assert np.linalg.norm(historical_after - historical_before) > 1e-5


def test_clone_preserves_optimized_values_and_marginals():
    smoother = initialized()
    smoother.add_observation(0, 0, 0, .3, 4, "obs0")
    smoother.add_odometry(0, 1, [.9, 0, .05], "odo1")
    smoother.add_observation(0, 1, 0, .4, 3, "obs1")
    smoother.optimize()
    duplicate = smoother.clone()
    np.testing.assert_array_equal(smoother.pose(0, 1), duplicate.pose(0, 1))
    np.testing.assert_array_equal(smoother.pose_covariance(0, 1), duplicate.pose_covariance(0, 1))
    assert duplicate.export_factors() == smoother.export_factors()


def test_simulator_packet_range_bearing_order_and_variances():
    smoother = GTSAMSmoother(backend="direct")
    smoother.add_measurement({"kind": "prior", "measurement_id": "prior", "to_pose": [0, 0],
                              "value": [0, 0, 0], "covariance": [.01, .01, .001]})
    smoother.add_measurement({"kind": "landmark", "measurement_id": "obs", "to_pose": [0, 0],
                              "landmark_id": 3, "value": [2, np.pi / 2], "covariance": [.04, .0025]})
    np.testing.assert_allclose(smoother.landmarks()[3], [0, 2], atol=1e-9)
    assert smoother.export_factors()[-1]["sigmas"] == [.05, .2]


def test_persistent_subprocess_matches_direct_and_survives_multiple_requests():
    with GTSAMSmoother(backend="subprocess", python_executable=sys.executable) as remote:
        remote.add_prior(0, [0, 0, 0], "prior")
        remote.add_observation(0, 0, 1, .2, 2, "obs")
        remote.add_odometry(0, 1, [1, 0, .1], "odom")
        direct = GTSAMSmoother(backend="direct")
        direct.merge_factors(remote.export_factors())
        np.testing.assert_allclose(remote.pose(0, 1), direct.pose(0, 1), atol=1e-12)
        np.testing.assert_allclose(remote.pose_covariance(0, 1), direct.pose_covariance(0, 1), atol=1e-12)
        assert remote.optimize()["factor_count"] == 3
