"""Keyed noisy Pose2/landmark simulator and fixed-frame evaluation metrics.

Truth lives in this simulator and the explicitly named evaluation accessors.
Estimator inputs are public initial-prior factors, noisy body-frame odometry,
known-ID range/bearing measurements, and optional noisy relative-pose factors.
No measurement contains a true landmark coordinate or world-frame displacement.
Units are metres and radians; poses use a common declared Cartesian gauge.
"""

from __future__ import annotations

from hashlib import sha256
import math
from typing import Mapping, Sequence

import numpy as np


def wrap_angle(angle):
    """Map angles to [-pi, pi), preserving scalar/array shape."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def compose_pose(pose, body_delta) -> np.ndarray:
    """SE(2) composition; translation in the robot's pre-motion body frame."""
    pose = _vector(pose, 3, "pose")
    delta = _vector(body_delta, 3, "body_delta")
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([pose[0] + c * delta[0] - s * delta[1], pose[1] + s * delta[0] + c * delta[1], float(wrap_angle(pose[2] + delta[2]))])


def between_poses(first, second) -> np.ndarray:
    """Pose2 of ``second`` expressed in ``first``'s body frame."""
    first = _vector(first, 3, "first")
    second = _vector(second, 3, "second")
    dx, dy = second[:2] - first[:2]
    c, s = np.cos(first[2]), np.sin(first[2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, float(wrap_angle(second[2] - first[2]))])


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result


def _sigmas(value, size, name):
    result = _vector(value, size, name)
    if np.any(result < 0):
        raise ValueError(f"{name} must contain nonnegative standard deviations")
    return result


def keyed_rng(seed: int, *key) -> np.random.Generator:
    """Stable keyed generator, independent of Python hash and call order."""
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    encoded = repr((int(seed), tuple(key))).encode("utf-8")
    entropy = int.from_bytes(sha256(encoded).digest()[:16], "little")
    return np.random.default_rng(entropy)


def generate_landmarks(seed: int, count: int = 6, bounds=(-1.0, 5.0, -1.0, 5.0)) -> dict[int, np.ndarray]:
    """Generate a simulator-only fixed map with its own independent map stream."""
    if count < 1:
        raise ValueError("count must be positive")
    xmin, xmax, ymin, ymax = _vector(bounds, 4, "bounds")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bounds must have increasing x/y limits")
    rng = keyed_rng(seed, "landmark_map_v1")
    xy = rng.uniform([xmin, ymin], [xmax, ymax], size=(count, 2))
    return {index: point.copy() for index, point in enumerate(xy)}


class LandmarkSimulator:
    """Body-frame stochastic motion with noisy known-ID landmark sensing.

    ``initial_poses`` are simulator truth, never an estimator observation.
    ``prior_means`` are the declared public means. If omitted, the public
    reset mean equals the declared starting pose, with the specified positive
    uncertainty; this is a known-start prior assumption, not exact localization.
    With ``sample_initial_prior=True``, truth is sampled around these public
    means using the separate prior stream, ignoring the supplied truth start.

    ``motion_std`` perturbs executed commands; ``odom_std`` perturbs the actual
    relative motion measurement. Thus odometry is noisy body-frame sensing,
    rather than commanded motion or privileged true world displacement.
    Sensor sigma order is (range metres, bearing radians). Covariance fields
    contain diagonal variances in the same order as each measurement's value.
    Keyed draws ensure repeated sensing at one timestamp yields the same ID
    and payload, allowing the estimator to reject duplicated information.
    """

    def __init__(
        self,
        initial_poses,
        landmarks: Mapping[int, Sequence[float]],
        *,
        motion_std=(.015, .015, .01),
        odom_std=(.05, .05, .02),
        sensor_std=(.1, .02),
        sensor_range=5.0,
        sensor_fov=2 * np.pi,
        relative_std=(.05, .05, .02),
        seeds: Mapping[str, int] | None = None,
        prior_means=None,
        prior_std=(.05, .05, .02),
        sample_initial_prior=False,
        scenario_id="landmark-fixture-v1",
    ):
        initial = np.asarray(initial_poses, dtype=np.float64)
        if initial.ndim != 2 or initial.shape[1] != 3 or not len(initial) or not np.isfinite(initial).all():
            raise ValueError("initial_poses must be a finite (robots,3) array")
        means = initial.copy() if prior_means is None else np.asarray(prior_means, dtype=np.float64)
        if means.shape != initial.shape or not np.isfinite(means).all():
            raise ValueError("prior_means must have the same finite shape as initial_poses")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ValueError("scenario_id must be a nonempty string")
        self.scenario_id = scenario_id
        self.n_robots = len(initial)
        self.motion_std = _sigmas(motion_std, 3, "motion_std")
        self.odom_std = _sigmas(odom_std, 3, "odom_std")
        self.sensor_std = _sigmas(sensor_std, 2, "sensor_std")
        self.relative_std = _sigmas(relative_std, 3, "relative_std")
        self.prior_std = _sigmas(prior_std, 3, "prior_std")
        if not np.isfinite(sensor_range) or sensor_range <= 0:
            raise ValueError("sensor_range must be positive")
        if not np.isfinite(sensor_fov) or not 0 < sensor_fov <= 2 * np.pi:
            raise ValueError("sensor_fov must lie in (0,2*pi]")
        self.sensor_range = float(sensor_range)
        self.sensor_fov = float(sensor_fov)
        self.seeds = {"motion": 1701, "odometry": 1702, "sensor": 1703, "relative": 1704, "prior": 1705}
        if seeds is not None:
            for name, value in seeds.items():
                if name not in self.seeds:
                    raise ValueError(f"unknown simulator seed stream {name}")
                if not isinstance(value, (int, np.integer)) or value < 0:
                    raise ValueError("simulator seeds must be nonnegative integers")
                self.seeds[name] = int(value)
        self._landmarks = {int(i): _vector(point, 2, "landmark").copy() for i, point in landmarks.items()}
        if not self._landmarks or len(self._landmarks) != len(landmarks):
            raise ValueError("landmarks must have unique integer IDs and finite positions")
        self.timestamp = 0
        self._prior_means = means.copy()
        self._poses = initial.copy()
        if sample_initial_prior:
            for robot in range(self.n_robots):
                perturbation = self._noise("prior", robot, 0, "initial", self.prior_std)
                self._poses[robot] = compose_pose(means[robot], perturbation)
        self._poses[:, 2] = wrap_angle(self._poses[:, 2])
        self._history = [self._poses.copy()]
        self.sensor_events: list[dict] = []
        self._seen_sensor_events = set()
        self.priors = [
            {
                "measurement_id": self._id("prior", robot, 0), "kind": "prior",
                "owner": robot, "timestamp": 0, "to_pose": [robot, 0],
                "value": means[robot].tolist(), "covariance": (self.prior_std ** 2).tolist(),
                "units": ["m", "m", "rad"], "gauge": "declared_common_world_frame",
                "covariance_frame": "pose2_local_tangent",
            }
            for robot in range(self.n_robots)
        ]

    def _id(self, family, robot, timestamp, *suffix):
        parts = [self.scenario_id, family, f"r{robot}", f"t{timestamp}"]
        parts.extend(str(value) for value in suffix)
        return ":".join(parts)

    def _noise(self, family, robot, timestamp, key, sigma):
        return keyed_rng(self.seeds[family], self.scenario_id, family, robot, timestamp, key).normal(size=len(sigma)) * sigma

    def _robot(self, robot):
        if not isinstance(robot, (int, np.integer)) or not 0 <= robot < self.n_robots:
            raise ValueError("invalid robot ID")
        return int(robot)

    def truth_poses(self) -> np.ndarray:
        """Evaluation-only copy of current true world poses."""
        return self._poses.copy()

    def truth_trajectory(self, robot: int) -> np.ndarray:
        """Evaluation-only fixed-world-frame trajectory including timestamp zero."""
        robot = self._robot(robot)
        return np.asarray(self._history)[:, robot, :].copy()

    def truth_landmarks(self) -> dict[int, np.ndarray]:
        """Evaluation-only map; never included in a measurement dictionary."""
        return {i: point.copy() for i, point in self._landmarks.items()}

    def observe(self) -> list[dict]:
        """Noisy range/bearing factors at the current timestamp, with stable IDs.

        Visibility is generated from simulator truth, as for a real sensor.
        A physically invalid negative sampled range is explicitly discarded
        and retained in ``sensor_events``; it is never clipped into a biased
        apparently valid independent observation.
        """
        observations = []
        for robot, pose in enumerate(self._poses):
            for landmark_id, point in sorted(self._landmarks.items()):
                difference = point - pose[:2]
                true_range = float(np.linalg.norm(difference))
                bearing = float(wrap_angle(math.atan2(difference[1], difference[0]) - pose[2]))
                if true_range <= 1e-9 or true_range > self.sensor_range or abs(bearing) > self.sensor_fov / 2:
                    continue
                measurement_id = self._id("landmark", robot, self.timestamp, landmark_id)
                noise = self._noise("sensor", robot, self.timestamp, landmark_id, self.sensor_std)
                measured_range = true_range + noise[0]
                if measured_range <= 0:
                    if measurement_id not in self._seen_sensor_events:
                        self.sensor_events.append({"measurement_id": measurement_id, "status": "discarded_negative_sampled_range", "sampled_range_m": float(measured_range)})
                        self._seen_sensor_events.add(measurement_id)
                    continue
                observations.append({
                    "measurement_id": measurement_id, "kind": "landmark", "owner": robot,
                    "timestamp": self.timestamp, "to_pose": [robot, self.timestamp],
                    "landmark_id": landmark_id,
                    "value": [float(measured_range), float(wrap_angle(bearing + noise[1]))],
                    "covariance": (self.sensor_std ** 2).tolist(), "units": ["m", "rad"],
                })
        return observations

    def step(self, commands) -> list[dict]:
        """Execute one body-frame command per robot and return only measurements."""
        commands = np.asarray(commands, dtype=np.float64)
        if commands.shape != (self.n_robots, 3) or not np.isfinite(commands).all():
            raise ValueError("commands must have finite shape (robots,3)")
        previous = self._poses.copy()
        self.timestamp += 1
        odometry = []
        for robot, command in enumerate(commands):
            executed = command + self._noise("motion", robot, self.timestamp, "execution", self.motion_std)
            self._poses[robot] = compose_pose(previous[robot], executed)
            body_delta = between_poses(previous[robot], self._poses[robot])
            measured = body_delta + self._noise("odometry", robot, self.timestamp, "measurement", self.odom_std)
            measured[2] = wrap_angle(measured[2])
            odometry.append({
                "measurement_id": self._id("odometry", robot, self.timestamp), "kind": "odometry",
                "owner": robot, "timestamp": self.timestamp,
                "from_pose": [robot, self.timestamp - 1], "to_pose": [robot, self.timestamp],
                "value": measured.tolist(), "covariance": (self.odom_std ** 2).tolist(),
                "units": ["m_body", "m_body", "rad"],
            })
        self._history.append(self._poses.copy())
        return odometry + self.observe()

    def relative_measurement(self, robot_a: int, robot_b: int, *, max_range=None) -> dict | None:
        """Noisy full relative Pose2 sensing, distinct from a range-only sensor.

        This assumes a relative sensor capable of heading measurement. It is
        available only within the declared range, and its payload contains no
        absolute world pose or landmark location.
        """
        a, b = self._robot(robot_a), self._robot(robot_b)
        if a == b:
            raise ValueError("relative sensor needs two distinct robots")
        limit = self.sensor_range if max_range is None else float(max_range)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("relative sensor range must be positive")
        truth = between_poses(self._poses[a], self._poses[b])
        if np.linalg.norm(truth[:2]) > limit:
            return None
        value = truth + self._noise("relative", a, self.timestamp, b, self.relative_std)
        value[2] = wrap_angle(value[2])
        return {
            "measurement_id": self._id("relative_pose", a, self.timestamp, b), "kind": "relative_pose",
            "owner": a, "timestamp": self.timestamp, "from_pose": [a, self.timestamp],
            "to_pose": [b, self.timestamp], "value": value.tolist(),
            "covariance": (self.relative_std ** 2).tolist(), "units": ["m_body", "m_body", "rad"],
        }


def scripted_commands(kind: str, *, step_length=1.0, n_steps=8, side_steps=2) -> list[list[float]]:
    """Prespecified body-frame straight, square-loop, and out-and-back paths."""
    if not np.isfinite(step_length) or step_length <= 0 or n_steps < 1 or side_steps < 1:
        raise ValueError("trajectory sizes must be positive")
    forward = [float(step_length), 0.0, 0.0]
    if kind == "straight":
        return [forward.copy() for _ in range(n_steps)]
    if kind == "closed_loop":
        return [command for _ in range(4) for command in [*[forward.copy() for _ in range(side_steps)], [0.0, 0.0, np.pi / 2]]]
    if kind == "revisit":
        return [*[forward.copy() for _ in range(side_steps)], [0.0, 0.0, np.pi], *[forward.copy() for _ in range(side_steps)], [0.0, 0.0, np.pi]]
    raise ValueError("kind must be straight, closed_loop or revisit")


def trajectory_error_metrics(estimated, truth) -> dict:
    """Trajectory-wide translational ATE/RPE in one fixed frame, no alignment."""
    estimated, truth = np.asarray(estimated, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    if estimated.shape != truth.shape or truth.ndim != 2 or truth.shape[1] != 3 or len(truth) < 1 or not np.isfinite(estimated).all() or not np.isfinite(truth).all():
        raise ValueError("estimated and truth must have the same finite (timestamps,3) shape")
    translation_squared = np.sum((estimated[:, :2] - truth[:, :2]) ** 2, axis=1)
    heading_errors = wrap_angle(estimated[:, 2] - truth[:, 2])
    rpe_translation_squared, rpe_rotation_squared = [], []
    for index in range(1, len(truth)):
        expected = between_poses(truth[index - 1], truth[index])
        actual = between_poses(estimated[index - 1], estimated[index])
        relative_error = between_poses(expected, actual)
        rpe_translation_squared.append(float(np.sum(relative_error[:2] ** 2)))
        rpe_rotation_squared.append(float(relative_error[2] ** 2))
    return {
        "current_pose_translation_error_m": float(np.sqrt(translation_squared[-1])),
        "current_pose_heading_error_rad": float(abs(heading_errors[-1])),
        "trajectory_ate_translation_m": float(np.sqrt(np.mean(translation_squared))),
        "trajectory_heading_rmse_rad": float(np.sqrt(np.mean(heading_errors ** 2))),
        "trajectory_rpe_translation_m": float(np.sqrt(np.mean(rpe_translation_squared))) if rpe_translation_squared else None,
        "trajectory_rpe_rotation_rad": float(np.sqrt(np.mean(rpe_rotation_squared))) if rpe_rotation_squared else None,
        "trajectory_timestamps": len(truth), "rpe_pairs": len(rpe_translation_squared),
        "frame_convention": "fixed_declared_world_frame_no_alignment",
    }


def landmark_error_metrics(estimated: Mapping, truth: Mapping, landmark_ids: Sequence[int], *, missing_penalty_m=None) -> dict:
    """Landmark error over a prespecified ID set, retaining absent estimates.

    The primary fixed-ID RMSE is undefined until all required landmarks have
    estimates, unless a missing penalty was prespecified. An observed-only
    diagnostic is explicitly named and never substituted for that denominator.
    """
    ids = [int(i) for i in landmark_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("landmark_ids must be a nonempty fixed set without duplicates")
    estimated = {int(i): _vector(point, 2, "estimated landmark") for i, point in estimated.items()}
    truth = {int(i): _vector(point, 2, "truth landmark") for i, point in truth.items()}
    if any(i not in truth for i in ids):
        raise ValueError("truth must define every fixed landmark ID")
    if missing_penalty_m is not None and (not np.isfinite(missing_penalty_m) or missing_penalty_m < 0):
        raise ValueError("missing penalty must be finite and nonnegative")
    squared = {i: float(np.sum((estimated[i] - truth[i]) ** 2)) for i in ids if i in estimated}
    missing = [i for i in ids if i not in estimated]
    primary = None
    if not missing or missing_penalty_m is not None:
        total = sum(squared.values()) + len(missing) * float(missing_penalty_m or 0) ** 2
        primary = float(np.sqrt(total / len(ids)))
    return {
        "landmark_rmse_fixed_ids_m": primary,
        "landmark_rmse_observed_only_m": float(np.sqrt(np.mean(list(squared.values())))) if squared else None,
        "fixed_landmark_ids": ids, "missing_landmark_ids": missing,
        "fixed_landmark_count": len(ids), "estimated_landmark_count": len(squared),
        "landmark_estimate_coverage": len(squared) / len(ids),
        "missing_penalty_m": missing_penalty_m,
    }


def coverage_metrics(truth_trajectories, estimated_trajectories=None, *, bounds, resolution: float, sensor_range: float, sensor_fov=2 * np.pi) -> dict:
    """Physical/estimated sensor footprints over the same fixed free-plane grid.

    This obstacle-free synthetic visibility diagnostic is independent of
    landmark-estimation accuracy. Pose drift cannot increase physical coverage;
    estimated coverage is separately labeled and uses the same denominator.
    """
    xmin, xmax, ymin, ymax = _vector(bounds, 4, "bounds")
    if xmax <= xmin or ymax <= ymin or not np.isfinite(resolution) or resolution <= 0 or not np.isfinite(sensor_range) or sensor_range <= 0 or not 0 < sensor_fov <= 2 * np.pi:
        raise ValueError("coverage domain, resolution, range and fov must be valid")
    nx = int(np.ceil((xmax - xmin) / resolution))
    ny = int(np.ceil((ymax - ymin) / resolution))
    xleft = xmin + np.arange(nx) * resolution
    yleft = ymin + np.arange(ny) * resolution
    xs = (xleft + np.minimum(xleft + resolution, xmax)) / 2
    ys = (yleft + np.minimum(yleft + resolution, ymax)) / 2
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    def count_visible(trajectories):
        visible = np.zeros(len(points), dtype=bool)
        for trajectory in trajectories:
            poses = np.asarray(trajectory, dtype=np.float64)
            if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
                raise ValueError("coverage trajectories must contain finite Pose2 rows")
            for pose in poses:
                delta = points - pose[:2]
                angle = wrap_angle(np.arctan2(delta[:, 1], delta[:, 0]) - pose[2])
                visible |= (np.linalg.norm(delta, axis=1) <= sensor_range) & (np.abs(angle) <= sensor_fov / 2)
        return int(visible.sum())

    physical_count = count_visible(truth_trajectories)
    estimated_count = count_visible(estimated_trajectories) if estimated_trajectories is not None else None
    return {
        "physical_coverage": physical_count / len(points),
        "estimated_coverage": estimated_count / len(points) if estimated_count is not None else None,
        "physical_visible_cells": physical_count, "estimated_visible_cells": estimated_count,
        "fixed_domain_cells": len(points), "bounds": [float(x) for x in bounds],
        "resolution_m": resolution, "coverage_model": "obstacle_free_cell_center_sensor_visibility",
    }
