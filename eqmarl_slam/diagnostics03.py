"""Genuine Pose2/landmark SLAM fixtures and fixed-frame evaluation records."""

from __future__ import annotations

import json
from pathlib import Path
import resource
import time

import numpy as np


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def fixed_frame_metrics(truth, estimated):
    """Trajectory ATE and one-step RPE with no fitted truth alignment.

    Translation ATE is RMS Euclidean position error over every retained pose,
    including the initial pose. RPE compares consecutive body-frame motions.
    Current error is separately reported and never called trajectory ATE.
    """
    truth, estimated = np.asarray(truth, dtype=float), np.asarray(estimated, dtype=float)
    if truth.ndim != 2 or truth.shape != estimated.shape or truth.shape[1] != 3 or len(truth) < 1:
        raise ValueError("Truth and estimates must have the same nonempty (time,3) shape")
    if not np.isfinite(truth).all() or not np.isfinite(estimated).all():
        raise ValueError("Nonfinite trajectory")
    error = estimated - truth
    error[:, 2] = _wrap(error[:, 2])

    def motions(poses):
        world = np.diff(poses[:, :2], axis=0)
        heading = poses[:-1, 2]
        return np.column_stack([np.cos(heading) * world[:, 0] + np.sin(heading) * world[:, 1],
                                -np.sin(heading) * world[:, 0] + np.cos(heading) * world[:, 1],
                                _wrap(np.diff(poses[:, 2]))])

    relative_error = motions(estimated) - motions(truth)
    if len(relative_error):
        relative_error[:, 2] = _wrap(relative_error[:, 2])
    return {"frame": "fixed_common_prior_frame_no_alignment", "poses": len(truth),
            "translation_ate_rmse_m": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
            "heading_rmse_rad": float(np.sqrt(np.mean(error[:, 2] ** 2))),
            "translation_rpe_rmse_m": float(np.sqrt(np.mean(np.sum(relative_error[:, :2] ** 2, axis=1)))) if len(relative_error) else None,
            "heading_rpe_rmse_rad": float(np.sqrt(np.mean(relative_error[:, 2] ** 2))) if len(relative_error) else None,
            "current_translation_error_m": float(np.linalg.norm(error[-1, :2])),
            "current_heading_error_rad": float(abs(error[-1, 2]))}


def fixed_landmark_metrics(truth_landmarks, estimates):
    truth = {int(key): np.asarray(value) for key, value in truth_landmarks.items()}
    estimates = {int(key): np.asarray(value) for key, value in estimates.items()}
    observed = sorted(set(truth) & set(estimates))
    missing = sorted(set(truth) - set(estimates))
    square_error = sum(float(np.sum((estimates[key] - truth[key]) ** 2)) for key in observed)
    return {"fixed_landmark_ids": sorted(truth), "landmark_count": len(truth),
            "observed_landmarks": len(observed), "missing_landmark_ids": missing,
            "landmark_recall": len(observed) / len(truth) if truth else 1.0,
            "observed_landmark_rmse_m": float(np.sqrt(square_error / len(observed))) if observed else None,
            "fixed_domain_landmark_rmse_m": float(np.sqrt(square_error / len(truth))) if truth and not missing else None,
            "frame": "fixed_common_prior_frame_no_alignment"}


def covariance_summary(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("Expected a finite 3x3 pose covariance")
    sign, logdet = np.linalg.slogdet(covariance)
    minimum = float(np.min(np.linalg.eigvalsh((covariance + covariance.T) / 2)))
    return {"trace": float(np.trace(covariance)), "xy_trace_m2": float(np.trace(covariance[:2, :2])),
            "logdet": float(logdet), "minimum_eigenvalue": minimum,
            "positive_definite": bool(sign > 0 and minimum > 0),
            "symmetric": bool(np.allclose(covariance, covariance.T, atol=1e-10)),
            "convention": "world_xy_heading; fixed m/rad units; same-dimensional comparisons only"}


def _cpu_seconds():
    own, child = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + child.ru_utime + child.ru_stime


def _new_smoother(config):
    from .slam import GTSAMSmoother

    return GTSAMSmoother(**{key: config[key] for key in (
        "prior_sigmas", "odometry_sigmas", "observation_sigmas", "relative_sigmas", "backend")})


def run_diagnostics(config, output_dir, emit, progress=None):
    """Archive scripted sensors, full-history posteriors, and controlled gates."""
    from .simulation import LandmarkSimulator, scripted_commands

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started, cpu_started = time.perf_counter(), _cpu_seconds()
    resources = {"primitive_steps": 0, "decisions": 0, "updates": 0, "wall_seconds": 0.0,
                 "cpu_core_hours": 0.0, "gpu_device_hours": 0, "circuit_evaluations": 0, "shots": 0,
                 "solver_requests": 0, "communication_bytes": 0}
    if progress is not None:
        progress["resources"] = resources
    gates, summaries, artifacts = {}, [], []
    max_wall = min(3600.0, float(config["wall_seconds"]))
    limits = config["gates"]
    measurement_path = output_dir / "measurements.jsonl"
    measurement_file = measurement_path.open("x")
    artifacts.append(measurement_path.name)

    def record(row):
        resources["wall_seconds"] = time.perf_counter() - started
        resources["cpu_core_hours"] = (_cpu_seconds() - cpu_started) / 3600
        if resources["wall_seconds"] >= max_wall:
            raise TimeoutError("Stage 03 diagnostic family exhausted its wall-time cap")
        if resources["primitive_steps"] > int(config["max_primitive_steps"]):
            raise TimeoutError("Stage 03 diagnostic family exhausted its primitive-step cap")
        emit(_jsonable(row))

    def check(name, passed, **details):
        gates[name] = bool(passed)
        record({"kind": "correctness_fixture", "fixture": name, "passed": bool(passed), **details})

    def optimize(graph):
        result = graph.optimize()
        resources["solver_requests"] += 1
        record({"kind": "solver_request", **result,
                "counter_convention": "Requests include cached no-ops; optimization_count is backend cumulative for this graph."})

    def ingest(graph, measurement, *, scenario):
        added = graph.add_measurement(measurement)
        raw = {"kind": "measurement", "scenario_id": scenario,
               "measurement": measurement, "accepted_new_factor": bool(added)}
        measurement_file.write(json.dumps(_jsonable(raw)) + "\n")
        measurement_file.flush()
        record(raw)
        return added

    def direct(graph, method, *args, scenario, **kwargs):
        added = getattr(graph, method)(*args, **kwargs)
        if added:
            raw_factor = graph.export_factors()[-1]
            raw = {"kind": "measurement", "scenario_id": scenario,
                   "measurement": raw_factor, "accepted_new_factor": True,
                   "measurement_source": "controlled_fixture_not_noisy_script"}
            measurement_file.write(json.dumps(_jsonable(raw)) + "\n")
            measurement_file.flush()
            record(raw)
        return added

    def snapshot(graph, robot, step):
        covariance = np.asarray(graph.pose_covariance(robot, step, world_frame=True))
        trajectory = graph.trajectory(robot)
        return {"robot": robot, "step": step, "pose": np.asarray(graph.pose(robot, step)),
                "pose_covariance_world": covariance, "uncertainty": covariance_summary(covariance),
                "trajectory": {str(key): np.asarray(value) for key, value in trajectory.items()},
                "landmarks": graph.landmarks(), "factor_count": graph.factor_count}

    def checkpoint(graph, scenario, robot_steps):
        name = f"{scenario}_graph.json"
        path = output_dir / name
        package = {"schema_version": "slam-raw-factors-v1", "scenario_id": scenario,
                   "backend": config["backend"], "coordinate_convention": config["pose_covariance"],
                   "smoother_configuration": {key: config[key] for key in (
                       "prior_sigmas", "odometry_sigmas", "observation_sigmas", "relative_sigmas", "backend")},
                   "factors": graph.export_factors(), "robot_steps": robot_steps}
        path.write_text(json.dumps(_jsonable(package), indent=2) + "\n")
        artifacts.append(name)
        # Rebuild from bytes read from the checkpoint, not the live graph.
        saved = json.loads(path.read_text())
        replay = _new_smoother(config)
        replay.merge_factors(saved["factors"])
        optimize(replay)
        pose_error, covariance_error = 0.0, 0.0
        for robot, steps in robot_steps:
            for step in steps:
                pose_error = max(pose_error, float(np.max(np.abs(np.asarray(graph.pose(robot, step)) - replay.pose(robot, step)))))
                covariance_error = max(covariance_error, float(np.max(np.abs(np.asarray(graph.pose_covariance(robot, step)) - replay.pose_covariance(robot, step)))))
        tolerance = float(limits["replay_pose_covariance_atol"])
        passed = replay.factor_count == graph.factor_count and max(pose_error, covariance_error) <= tolerance
        check(f"replay_{scenario}", passed, source_checkpoint=name, factor_count=replay.factor_count,
              pose_max_absolute_error=pose_error, covariance_max_absolute_error=covariance_error)

    try:
        # One-robot noisy straight and closed-loop paths are assessed before
        # cooperation; sensor outputs, never truth poses, enter the smoother.
        truth_landmarks = {int(key): value for key, value in config["landmarks"].items()}
        for script in config.get("script_names", ["straight", "closed_loop"]):
            scenario = f"{script.replace('_', '-')}-script-v1"
            sim = LandmarkSimulator(config["initial_poses"], truth_landmarks,
                                   motion_std=config["motion_std"], odom_std=config["odom_std"],
                                   sensor_std=config["sensor_std"], sensor_range=config["sensor_range"],
                                   sensor_fov=config["sensor_fov"], relative_std=config["relative_sigmas"],
                                   seeds={"motion": config["motion_seed"], "odometry": config["odometry_seed"],
                                          "sensor": config["sensor_seed"], "relative": config["relative_seed"],
                                          "prior": config["prior_seed"]}, scenario_id=scenario,
                                   prior_std=config["prior_sigmas"], sample_initial_prior=config["sample_initial_prior"])
            graph = _new_smoother(config)
            for measurement in sim.priors:
                ingest(graph, measurement, scenario=scenario)
            for measurement in sim.observe():
                ingest(graph, measurement, scenario=scenario)
            optimize(graph)
            record({"kind": "initial_belief", "scenario_id": scenario, "posterior": snapshot(graph, 0, 0),
                    "evaluation_only_truth_pose": sim.truth_trajectory(0)[0]})
            commands = scripted_commands(script, step_length=float(config["script_step_length"]),
                                         n_steps=int(config["script_n_steps"]), side_steps=int(config["script_side_steps"]))
            for step, command in enumerate(commands, 1):
                measurements = sim.step([command])
                resources["primitive_steps"] += 1
                odometry = [measurement for measurement in measurements if measurement["kind"] == "odometry"]
                observations = [measurement for measurement in measurements if measurement["kind"] == "landmark"]
                if len(odometry) != 1 or len(odometry) + len(observations) != len(measurements):
                    raise ValueError("Unexpected simulator factor contract")
                for measurement in odometry:
                    ingest(graph, measurement, scenario=scenario)
                optimize(graph)
                prediction = snapshot(graph, 0, step)
                for measurement in observations:
                    ingest(graph, measurement, scenario=scenario)
                optimize(graph)
                posterior = snapshot(graph, 0, step)
                truth = sim.truth_trajectory(0)
                estimated = np.array([graph.pose(0, index) for index in range(step + 1)])
                metrics = fixed_frame_metrics(truth, estimated)
                error = np.asarray(posterior["pose"]) - truth[-1]
                error[2] = _wrap(error[2])
                record({"kind": "slam_step", "scenario_id": scenario, "step": step,
                        "body_command": command, "prediction": prediction, "posterior": posterior,
                        "evaluation_only_truth_trajectory": truth, "evaluation_only_truth_landmarks": sim.truth_landmarks(),
                        "trajectory_metrics": metrics,
                        "current_pose_nees_diagnostic": float(error @ np.linalg.solve(posterior["pose_covariance_world"], error)),
                        "nees_interpretation": "single-trajectory diagnostic, not a covariance calibration study"})
                if not (prediction["uncertainty"]["positive_definite"] and posterior["uncertainty"]["positive_definite"]):
                    check(f"{script}_finite_positive_covariance", False, step=step)
            final_metrics = fixed_frame_metrics(sim.truth_trajectory(0), np.array([graph.pose(0, index) for index in range(len(commands) + 1)]))
            final_metrics.update(fixed_landmark_metrics(sim.truth_landmarks(), graph.landmarks()))
            summaries.append({"scenario_id": scenario, **final_metrics})
            check(f"{script}_finite_positive_covariance", gates.get(f"{script}_finite_positive_covariance", True)
                  and snapshot(graph, 0, len(commands))["uncertainty"]["positive_definite"], metrics=final_metrics)
            checkpoint(graph, scenario, [(0, list(range(len(commands) + 1)))])

        # A new unknown landmark adds a map variable; its first bearing/range
        # observation alone must not manufacture independent pose information.
        scenario = "first-unknown-landmark-v1"
        graph = _new_smoother(config)
        direct(graph, "add_prior", 0, [0, 0, 0], "first/prior", scenario=scenario)
        direct(graph, "add_odometry", 0, 1, [1, 0, 0], "first/odom", scenario=scenario)
        resources["primitive_steps"] += 1
        optimize(graph)
        before = snapshot(graph, 0, 1)
        direct(graph, "add_observation", 0, 1, 91, 0.4, 2.0, "first/landmark", scenario=scenario)
        optimize(graph)
        after = snapshot(graph, 0, 1)
        new_landmark_error = float(np.max(np.abs(before["pose_covariance_world"] - after["pose_covariance_world"])))
        check("first_unknown_landmark_no_pose_information", new_landmark_error <= limits["new_unknown_landmark_pose_covariance_atol"],
              prediction=before, posterior=after, covariance_max_absolute_difference=new_landmark_error)

        # Controlled drift is deliberately confined to this named fixture.
        # The landmark is initially unknown; a later independent revisit
        # constrains the same map variable and revises historical poses.
        scenario = "controlled-biased-square-closure-v1"
        graph = _new_smoother(config)
        direct(graph, "add_prior", 0, [0, 0, 0], "closure/prior", scenario=scenario)
        landmark = np.asarray(truth_landmarks[0])
        initial_bearing, initial_range = float(np.arctan2(landmark[1], landmark[0])), float(np.linalg.norm(landmark))
        direct(graph, "add_observation", 0, 0, 0, initial_bearing, initial_range, "closure/initial-landmark", scenario=scenario)
        optimize(graph)
        closure = config["controlled_closure"]
        commands = np.asarray(closure["body_commands"], dtype=float)
        bias = np.asarray(closure["odometry_bias"], dtype=float)
        truth = [np.zeros(3)]
        for step, command in enumerate(commands, 1):
            truth.append(truth[-1] + command)
            direct(graph, "add_odometry", 0, step, (command + bias).tolist(), f"closure/odom/{step}", scenario=scenario)
            resources["primitive_steps"] += 1
        optimize(graph)
        prediction = snapshot(graph, 0, len(commands))
        predicted_history = np.array([graph.pose(0, step) for step in range(len(commands) + 1)])
        final_truth = truth[-1]
        delta = landmark - final_truth[:2]
        bearing, distance = float(_wrap(np.arctan2(delta[1], delta[0]) - final_truth[2])), float(np.linalg.norm(delta))
        direct(graph, "add_observation", 0, len(commands), 0, bearing, distance, "closure/revisit", scenario=scenario)
        optimize(graph)
        posterior = snapshot(graph, 0, len(commands))
        posterior_history = np.array([graph.pose(0, step) for step in range(len(commands) + 1)])
        predicted_metrics, posterior_metrics = fixed_frame_metrics(truth, predicted_history), fixed_frame_metrics(truth, posterior_history)
        trace_gain = prediction["uncertainty"]["trace"] - posterior["uncertainty"]["trace"]
        logdet_gain = prediction["uncertainty"]["logdet"] - posterior["uncertainty"]["logdet"]
        historical_correction = float(np.linalg.norm(posterior_history[1:-1] - predicted_history[1:-1]))
        check("same_step_known_landmark_revisit", trace_gain > limits["same_step_trace_reduction_min"]
              and logdet_gain > limits["same_step_logdet_reduction_min"], prediction=prediction, posterior=posterior,
              trace_reduction=trace_gain, logdet_reduction=logdet_gain,
              comparison="same timestamp and same 3x3 world-frame marginal; no new landmark IDs")
        check("historical_pose_correction", historical_correction > limits["historical_pose_correction_min"]
              and (not limits["controlled_closure_ate_must_reduce"] or posterior_metrics["translation_ate_rmse_m"] < predicted_metrics["translation_ate_rmse_m"]),
              historical_correction_norm=historical_correction, prediction_metrics=predicted_metrics,
              posterior_metrics=posterior_metrics, evaluation_only_truth_trajectory=truth)

        count = graph.factor_count
        duplicate_added = direct(graph, "add_observation", 0, len(commands), 0, bearing, distance, "closure/revisit", scenario=scenario)
        optimize(graph)
        duplicate_error = float(np.max(np.abs(np.asarray(graph.pose_covariance(0, len(commands))) - posterior["pose_covariance_world"])))
        conflicting_rejected = False
        try:
            graph.add_observation(0, len(commands), 0, bearing + 0.1, distance, "closure/revisit")
        except ValueError:
            conflicting_rejected = True
        check("measurement_id_deduplication", not duplicate_added and graph.factor_count == count
              and duplicate_error <= limits["duplicate_covariance_atol"] and conflicting_rejected,
              duplicate_added=bool(duplicate_added), factor_count_before=count, factor_count_after=graph.factor_count,
              covariance_max_absolute_difference=duplicate_error, conflicting_id_rejected=conflicting_rejected)
        checkpoint(graph, scenario, [(0, list(range(len(commands) + 1)))])

        scenario = "relative-robot-v1"
        graph = _new_smoother(config)
        for robot, prior in [(0, [0, 0, 0]), (1, [0, 2, 0])]:
            direct(graph, "add_prior", robot, prior, f"relative/{robot}/prior", scenario=scenario)
            direct(graph, "add_odometry", robot, 1, [1, 0, 0], f"relative/{robot}/odom", scenario=scenario)
        resources["primitive_steps"] += 1
        optimize(graph)
        before = snapshot(graph, 0, 1)
        direct(graph, "add_relative", 0, 1, 1, 1, [0, 2, 0], "relative/measurement", scenario=scenario)
        optimize(graph)
        after = snapshot(graph, 0, 1)
        relative_gain = before["uncertainty"]["trace"] - after["uncertainty"]["trace"]
        check("relative_robot_information", relative_gain > limits["relative_information_gain_min"],
              prediction=before, posterior=after, trace_reduction=relative_gain,
              assumptions="declared common reference priors; measured relative Pose2, not shared true poses")
        checkpoint(graph, scenario, [(0, [0, 1]), (1, [0, 1])])

        scenario = "factor-delivery-v1"
        local = []
        for robot, prior, measured_bearing in [(0, [0, 0, 0], np.pi / 4), (1, [0, 2, 0], -np.pi / 4)]:
            owned = _new_smoother(config)
            direct(owned, "add_prior", robot, prior, f"local/{robot}/prior", scenario=scenario)
            direct(owned, "add_odometry", robot, 1, [1, 0, 0], f"local/{robot}/odom", scenario=scenario)
            direct(owned, "add_observation", robot, 1, 0, float(measured_bearing), float(np.sqrt(2)), f"local/{robot}/landmark", scenario=scenario)
            optimize(owned)
            local.append(owned)
        resources["primitive_steps"] += 1
        private_before = snapshot(local[0], 0, 1)
        direct(local[1], "add_odometry", 1, 2, [0.2, 0, 0], "local/1/private-next", scenario=scenario)
        optimize(local[1])
        resources["primitive_steps"] += 1
        private_after = snapshot(local[0], 0, 1)
        private_error = float(np.max(np.abs(private_before["pose_covariance_world"] - private_after["pose_covariance_world"])))
        packet = json.dumps(_jsonable(local[1].export_factors()), sort_keys=True, separators=(",", ":")).encode()
        packet_path = output_dir / "delivered_factor_packet.json"
        packet_path.write_bytes(packet)
        artifacts.append(packet_path.name)
        shared = local[0].clone()
        received = json.loads(packet_path.read_text())
        new_factors = shared.merge_factors(received)
        resources["communication_bytes"] += len(packet)
        optimize(shared)
        shared_after = snapshot(shared, 0, 1)
        shared_gain = private_after["uncertainty"]["trace"] - shared_after["uncertainty"]["trace"]
        shared_count = shared.factor_count
        duplicate_merge = shared.merge_factors(received)
        optimize(shared)
        check("local_owner_and_delivered_factor_boundary", private_error <= limits["duplicate_covariance_atol"]
              and local[0].factor_count == private_before["factor_count"] and new_factors > 0
              and shared_gain > limits["relative_information_gain_min"] and duplicate_merge == 0
              and shared.factor_count == shared_count and len(packet) > 0,
              private_prediction=private_before, private_after_teammate_update=private_after,
              shared_posterior=shared_after, private_covariance_max_absolute_difference=private_error,
              shared_trace_reduction=shared_gain, delivered_payload_bytes=len(packet),
              imported_new_factors=new_factors, duplicate_delivery_new_factors=duplicate_merge,
              payload_convention="UTF8 compact JSON full raw-factor packet, including IDs and noise; retransmission bytes charged separately")
        # A second delivery consumes bytes even when no new factor is accepted.
        resources["communication_bytes"] += len(packet)
        record({"kind": "communication", "scenario_id": scenario, "deliveries": 2,
                "bytes_per_delivery": len(packet), "communication_bytes": 2 * len(packet),
                "new_factors_per_delivery": [new_factors, duplicate_merge]})
        checkpoint(shared, scenario, [(0, [0, 1]), (1, [0, 1, 2])])
        inventories = {"last_graph_factors": shared.export_factors(), "total_communication_bytes": resources["communication_bytes"],
                       "owner_rule": "only explicit packet ingestion adds teammate factors; canonical measurement IDs deduplicate raw evidence"}
        inventory_path = output_dir / "factor_inventory.json"
        inventory_path.write_text(json.dumps(_jsonable(inventories), indent=2) + "\n")
        artifacts.append(inventory_path.name)
    finally:
        measurement_file.close()
    resources["wall_seconds"] = time.perf_counter() - started
    resources["cpu_core_hours"] = (_cpu_seconds() - cpu_started) / 3600
    return {"passed": all(gates.values()), "gates": gates, "resources": resources,
            "summary": {"scenarios": summaries, "backend": config["backend"],
                        "evaluation_frame": "fixed_common_prior_frame_no_alignment", "training": False,
                        "estimator_kind": "full-history Pose2/unknown-Point2 factor-graph smoother",
                        "state_source": "sensor factors and declared uncertain priors; truth is evaluation-only",
                        "primitive_step_unit": "joint simulated motion/controlled odometry timestamp; no actor or learner updates",
                        "covariance_interpretation": "Gaussian linearization marginal, tested under registered measurement/association model; NEES rows are diagnostics, not calibration evidence"},
            "required_artifacts": artifacts}
