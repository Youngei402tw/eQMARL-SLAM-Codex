"""Stage 02 fixed correctness cases and registered exploration diagnostics.

Policies receive only observation dictionaries. Truth-derived fields are kept
in explicitly evaluation-only output records and never passed to the planner.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np


DEFAULT_SCENARIOS = [
    {"scenario_id": "open", "grid": ["#########"] + ["#.......#"] * 7 + ["#########"],
     "starts": [[1, 1], [7, 7]]},
    {"scenario_id": "bottleneck", "grid": ["###########", "#....#....#", "#....#....#",
     "#.........#", "#....#....#", "#....#....#", "###########"], "starts": [[1, 1], [5, 9]]},
    {"scenario_id": "deadend", "grid": ["###########", "#.........#", "#####.#####",
     "#.....#...#", "#.###.#.#.#", "#...#...#.#", "###########"], "starts": [[1, 1], [5, 9]]},
]
METHODS = ("random", "reactive", "local_frontier", "shared_frontier")


def scenario_specs(config):
    """Return a JSON copy, preventing an environment from changing preregistration."""
    scenarios = json.loads(json.dumps(config.get("scenarios", DEFAULT_SCENARIOS)))
    if not scenarios or len({scenario["scenario_id"] for scenario in scenarios}) != len(scenarios):
        raise ValueError("Scenarios require unique explicit scenario_id values")
    for scenario in scenarios:
        rows = scenario["grid"]
        if not rows or len({len(row) for row in rows}) != 1 or len(scenario["starts"]) != 2:
            raise ValueError("Stage 02 requires rectangular fixed maps and two explicit starts")
    return scenarios


def _env_config(config, *, shared=False, **overrides):
    result = {"sensor_range": int(config.get("sensor_range", 2)),
              "horizon": int(config.get("horizon", 80)),
              "n_agents": int(config.get("n_agents", 2)),
              "coverage_target": float(config.get("coverage_target", 1.0)),
              "information_condition": "shared-map" if shared else "local-only"}
    if "reward" in config:
        result["reward"] = dict(config["reward"])
    result.update(overrides)
    return result


def _equal_observations(before, after):
    return before.keys() == after.keys() and all(np.array_equal(before[key], after[key]) for key in before)


def correctness_fixtures(config, emit, *, resources=None):
    """Small deterministic falsification cases, also suitable for unit checks."""
    from .grid import GridExplorationEnv
    from .planning import action_for_path, bfs_path, greedy_local_frontier

    gates, rows, primitive_steps = {}, [], 0

    def transition(env, actions):
        nonlocal primitive_steps
        primitive_steps += 1
        if resources is not None:
            resources["primitive_steps"] = primitive_steps
            resources["decisions"] = 2 * primitive_steps
        return env.step(np.asarray(actions, dtype=np.int64))

    def check(name, passed, **details):
        row = {"kind": "correctness_fixture", "fixture": name, "passed": bool(passed), **details}
        gates[name] = bool(passed)
        rows.append(row)
        emit(row)

    corridor = {"scenario_id": "fixture-corridor", "grid": ["#######", "#.....#", "#######"],
                "starts": [[1, 1], [1, 5]]}
    env = GridExplorationEnv(_env_config(config, sensor_range=1, horizon=20), corridor)
    env.reset(seed=101)
    before = env.metrics()
    _, _, terminated, truncated, _ = transition(env, [2, 4])
    after = env.metrics()
    check("corridor_physical_coverage", np.isclose(before["physical_coverage"], 4 / 5)
          and np.isclose(after["physical_coverage"], 1) and terminated and not truncated,
          before=before, after=after, actions=[2, 4])

    # A wall seen by one robot must occlude the cells behind it, irrespective
    # of visibility supplied to the other robot.
    wall = {"scenario_id": "fixture-wall", "grid": ["#######", "#..#..#", "#######"],
            "starts": [[1, 1], [1, 5]]}
    env = GridExplorationEnv(_env_config(config, sensor_range=8), wall)
    env.reset(seed=102)
    own_map = env.observe_agent(0)["map"]
    check("wall_occlusion", own_map[1, 3] == 1 and own_map[1, 4] == -1,
          wall_value=int(own_map[1, 3]), occluded_cell_value=int(own_map[1, 4]))

    # The path planner must not enter unknown cells or return a path through a
    # wall. A current-cell target is an explicit wait, and selecting a frontier
    # skips that target when a reachable alternative exists.
    known = np.array([[0, 1, 0], [-1, 1, -1]], dtype=np.int8)
    unreachable = bfs_path(known, (0, 0), (0, 2))
    current = bfs_path(known, (0, 0), (0, 0))
    small = np.array([[-1, -1, -1], [-1, 0, 0], [-1, -1, -1]], dtype=np.int8)
    frontier_action = int(greedy_local_frontier(small, (1, 1), np.ones(5, dtype=bool)))
    check("unreachable_and_current_target", unreachable is None and current == [(0, 0)]
          and action_for_path(current) == 0 and frontier_action == 2,
          unreachable_path=unreachable, current_path=current, frontier_action=frontier_action)

    for name, scenario, actions in [
        ("wall_collision", corridor, [4, 2]),
        ("same_target_collision", {"scenario_id": "fixture-same-target", "grid": ["#####", "#...#", "#####"], "starts": [[1, 1], [1, 3]]}, [2, 4]),
        ("swap_collision", {"scenario_id": "fixture-swap", "grid": ["#####", "#...#", "#####"], "starts": [[1, 1], [1, 2]]}, [2, 4]),
    ]:
        env = GridExplorationEnv(_env_config(config, sensor_range=1), scenario)
        env.reset(seed=103)
        positions = env.positions.copy()
        transition(env, actions)
        metrics = env.metrics()
        check(name, np.array_equal(positions, env.positions) and metrics["collisions_total"] == 2,
              actions=actions, positions_before=positions.tolist(), positions_after=env.positions.tolist(),
              collisions_total=metrics["collisions_total"])

    env = GridExplorationEnv(_env_config(config, sensor_range=1), corridor)
    env.reset(seed=104)
    observation = {key: value.copy() for key, value in env.observe_agent(0).items()}
    env.known_maps[1, 1, 3] = 0
    check("private_information_isolation", _equal_observations(observation, env.observe_agent(0))
          and env.metrics()["communication_bytes"] == 0,
          communication_bytes=env.metrics()["communication_bytes"])

    env = GridExplorationEnv(_env_config(config, shared=True, sensor_range=1), corridor)
    env.reset(seed=105)
    observation = {key: value.copy() for key, value in env.observe_agent(0).items()}
    initial_bytes = env.metrics()["communication_bytes"]
    env.known_maps[1, 1, 3] = 0
    unchanged_before_delivery = _equal_observations(observation, env.observe_agent(0))
    env.deliver_shared_messages()
    refreshed = env.observe_agent(0)
    final_bytes = env.metrics()["communication_bytes"]
    check("shared_information_charged", unchanged_before_delivery and refreshed["map"][1, 3] == 0
          and not _equal_observations(observation, refreshed) and final_bytes > initial_bytes > 0,
          unchanged_before_delivery=unchanged_before_delivery,
          communication_bytes_before=initial_bytes, communication_bytes_after=final_bytes)

    env = GridExplorationEnv(_env_config(config, sensor_range=1), corridor)
    env.reset(seed=106)
    before = env.metrics()
    env.known_maps[:, env.reachable_free] = 0
    after = env.metrics()
    check("physical_coverage_independent_of_mapper", after["physical_coverage"] == before["physical_coverage"]
          and after["estimated_coverage"] > before["estimated_coverage"]
          and after["domain_sha256"] == before["domain_sha256"]
          and after["domain_cell_count"] == before["domain_cell_count"], before=before, after=after)

    env = GridExplorationEnv(_env_config(config, sensor_range=1, horizon=2), corridor)
    env.reset(seed=107)
    budgets = [float(env.observe_agent(0)["remaining_budget"][0])]
    _, _, first_terminal, first_truncated, _ = transition(env, [0, 0])
    budgets.append(float(env.observe_agent(0)["remaining_budget"][0]))
    _, _, last_terminal, last_truncated, _ = transition(env, [0, 0])
    budgets.append(float(env.observe_agent(0)["remaining_budget"][0]))
    check("intrinsic_horizon_budget", budgets == [1.0, 0.5, 0.0] and not first_terminal
          and last_terminal and not first_truncated and not last_truncated,
          remaining_budgets=budgets, terminated=bool(last_terminal), truncated=bool(last_truncated))

    env = GridExplorationEnv(_env_config(config, sensor_range=1, horizon=20), corridor)
    env.reset(seed=108)
    for _ in range(20):
        actions = [greedy_local_frontier(obs["map"], obs["pose"], obs["action_mask"])
                   for obs in [env.observe_agent(0), env.observe_agent(1)]]
        _, _, terminal, _, _ = transition(env, actions)
        if terminal:
            break
    check("corridor_frontier_complete", env.metrics()["physical_coverage"] == 1.0,
          final_metrics=env.metrics(), threshold=1.0)
    return {"passed": all(gates.values()), "gates": gates, "records": rows,
            "primitive_steps": primitive_steps}


def _planner_diagnostics(observations):
    from .planning import bfs_path, reachable_frontiers

    records = []
    for obs in observations:
        pose = tuple(int(value) for value in obs["pose"])
        candidates = [target for target in reachable_frontiers(obs["map"], pose) if tuple(target) != pose]
        paths = [bfs_path(obs["map"], pose, target) for target in candidates]
        lengths = [len(path) - 1 for path in paths if path is not None]
        records.append({"reachable_frontier_count": len(candidates),
                        "shortest_candidate_path_steps": min(lengths) if lengths else None})
    return records


def run_diagnostics(config, output_dir, emit, progress=None):
    """Run fixed fixtures then the committed known-pose development matrix."""
    from .grid import GridExplorationEnv
    from .planning import SharedFrontierAssignment, greedy_local_frontier, random_action, reactive_action

    started, cpu_started = time.perf_counter(), time.process_time()
    resources = {"primitive_steps": 0, "decisions": 0, "updates": 0,
                 "wall_seconds": 0.0, "cpu_core_hours": 0.0,
                 "gpu_device_hours": 0, "circuit_evaluations": 0, "shots": 0}
    if progress is not None:
        progress["resources"] = resources

    def measured_emit(row):
        resources["wall_seconds"] = time.perf_counter() - started
        resources["cpu_core_hours"] = (time.process_time() - cpu_started) / 3600
        emit(row)

    wall_cap = min(3600.0, float(config.get("wall_seconds", 3600)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed = correctness_fixtures(config, measured_emit, resources=resources)
    if not fixed["passed"]:
        return {"passed": False, "gates": fixed["gates"], "resources": resources,
                "summary": {"development_matrix_executed": False, "reason": "correctness fixture failed"}}
    scenarios = scenario_specs(config)
    seeds = [int(seed) for seed in config.get("development_seeds", [1101, 1102, 1103])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Development seeds must be nonempty, explicit, and unique")
    methods = config.get("methods", list(METHODS))
    if set(methods) != set(METHODS) or len(methods) != len(METHODS):
        raise ValueError("Stage 02 matrix requires all four registered control methods")
    expected_modes = {method: "sampled" if method == "random" else "greedy" for method in methods}
    if config.get("action_modes", expected_modes) != expected_modes:
        raise ValueError("Configured action modes differ from the registered controls")
    if config.get("known_pose", True) is not True or config.get("motion_noise", 0) != 0 or config.get("sensor_noise", 0) != 0:
        raise ValueError("Stage 02 controls require declared known pose and noiseless motion/sensors")
    manifest = {"protocol_id": config["protocol_id"], "scenario_kind": "fixed development maps",
                "scenarios": scenarios, "development_seeds": seeds,
                "train": None, "validation": None, "test": None,
                "policy_modes": {method: "sampled" if method == "random" else "greedy" for method in methods},
                "information_conditions": {method: "shared-map" if method == "shared_frontier" else "local-only" for method in methods},
                "environment_configs": {method: _env_config(config, shared=method == "shared_frontier") for method in methods},
                "seed_rule": "action_rng SeedSequence([development_seed,scenario_index,registered_method_index]); fixed maps and noiseless sensors paired across methods"}
    manifest_path = output_dir / "scenario_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    trace_path = output_dir / "trajectories.jsonl"
    episodes, primitive_steps = [], fixed["primitive_steps"]
    primitive_cap = int(config.get("max_primitive_steps", 3200))
    horizon = int(config.get("horizon", 80))
    with trace_path.open("x") as trace_file:
        for method in methods:
            for scenario_index, scenario in enumerate(scenarios):
                for seed in seeds:
                    if time.perf_counter() - started >= wall_cap:
                        raise TimeoutError("Stage 02 diagnostic family exhausted its wall-time cap")
                    env = GridExplorationEnv(_env_config(config, shared=method == "shared_frontier"), scenario)
                    env.reset(seed=seed)
                    initial = env.metrics()
                    action_seed = int(np.random.SeedSequence([seed, scenario_index, METHODS.index(method)]).generate_state(1)[0])
                    rng = np.random.default_rng(action_seed)
                    shared_policy = SharedFrontierAssignment() if method == "shared_frontier" else None
                    episode_id = f"{method}_{scenario['scenario_id']}_seed-{seed}"
                    measured_emit({"kind": "episode_start", "episode_id": episode_id, "method": method,
                                   "scenario_id": scenario["scenario_id"], "development_seed": seed,
                                   "action_seed": action_seed, "resolved_environment": env.config,
                                   "policy_mode": manifest["policy_modes"][method], "evaluation_only": initial})
                    coverage = [initial["physical_coverage"]]
                    episode_return = 0.0
                    targets = None
                    for step in range(1, horizon + 1):
                        if time.perf_counter() - started >= wall_cap:
                            raise TimeoutError("Stage 02 diagnostic family exhausted its wall-time cap")
                        if primitive_steps >= primitive_cap:
                            raise TimeoutError("Stage 02 diagnostic family exhausted its primitive-step cap")
                        observations = [env.observe_agent(agent) for agent in range(2)]
                        planner = _planner_diagnostics(observations)
                        if method == "random":
                            actions = [random_action(obs["action_mask"], rng=rng, action_mode="sampled") for obs in observations]
                        elif method == "reactive":
                            actions = [reactive_action(obs["map"], obs["pose"], obs["visited"], obs["action_mask"], rng=rng, action_mode="greedy")
                                       for obs in observations]
                        elif method == "local_frontier":
                            actions = [greedy_local_frontier(obs["map"], obs["pose"], obs["action_mask"]) for obs in observations]
                        else:
                            # Poses must be reconstructed from received messages,
                            # never from the environment's current position array.
                            delivered_poses = np.vstack([observations[0]["pose"], observations[0]["messages"]])
                            actions = shared_policy.actions(observations[0]["map"], delivered_poses,
                                                            [obs["action_mask"] for obs in observations])
                            targets = [None if target is None else [int(value) for value in target] for target in shared_policy.last_targets]
                        positions_before = env.positions.copy()
                        _, reward, terminated, truncated, info = env.step(np.asarray(actions, dtype=np.int64))
                        metrics = env.metrics()
                        primitive_steps += 1
                        resources["primitive_steps"] = primitive_steps
                        resources["decisions"] = 2 * primitive_steps
                        episode_return += float(reward)
                        coverage.append(metrics["physical_coverage"])
                        row = {"kind": "primitive_step", "episode_id": episode_id, "method": method,
                               "scenario_id": scenario["scenario_id"], "development_seed": seed, "action_seed": action_seed,
                               "step": step, "policy_mode": manifest["policy_modes"][method],
                               "information_condition": manifest["information_conditions"][method],
                               "actions": [int(action) for action in actions], "reward": float(reward),
                               "reward_components": info["evaluation_only"]["reward_components"],
                               "terminated": bool(terminated), "truncated": bool(truncated),
                               "planner": planner, "assigned_targets": targets,
                               "planner_progress": bool(np.any(positions_before != env.positions)
                                                        or coverage[-1] > coverage[-2]),
                               "evaluation_only": metrics}
                        measured_emit(row)
                        trace_file.write(json.dumps({**row, "evaluation_only_positions": env.positions.tolist()}) + "\n")
                        trace_file.flush()
                        if terminated or truncated:
                            break
                    # Fixed-budget AUC carries a terminal episode's final
                    # coverage forward; it does not invent extra transitions.
                    auc = (sum(coverage) + (horizon + 1 - len(coverage)) * coverage[-1]) / (horizon + 1)
                    episode = {"kind": "episode", "episode_id": episode_id, "method": method,
                               "scenario_id": scenario["scenario_id"], "development_seed": seed, "action_seed": action_seed,
                               "policy_mode": manifest["policy_modes"][method],
                               "information_condition": manifest["information_conditions"][method],
                               "initial_physical_coverage": initial["physical_coverage"],
                               "physical_coverage_auc": float(auc), "return": float(episode_return),
                               "primitive_steps": step, "budget_primitive_steps": horizon,
                               "terminated": bool(terminated), "truncated": bool(truncated),
                               **metrics}
                    episodes.append(episode)
                    measured_emit(episode)
                    if progress is not None:
                        progress["last_episode"] = {"episode_id": episode_id, "episodes_completed": len(episodes),
                                                    "physical_coverage": metrics["physical_coverage"]}
    groups = []
    for method in methods:
        for scenario in scenarios:
            matching = [row for row in episodes if row["method"] == method and row["scenario_id"] == scenario["scenario_id"]]
            groups.append({"method": method, "scenario_id": scenario["scenario_id"], "episodes": len(matching),
                           "mean_physical_coverage": float(np.mean([row["physical_coverage"] for row in matching])),
                           "mean_estimated_coverage": float(np.mean([row["estimated_coverage"] for row in matching])),
                           "mean_map_accuracy": float(np.mean([row["map_accuracy"] for row in matching])),
                           "mean_physical_coverage_auc": float(np.mean([row["physical_coverage_auc"] for row in matching])),
                           "successes": sum(bool(row["success"]) for row in matching)})
    gates = dict(fixed["gates"])
    open_ids = config.get("simple_progress_scenarios", ["open", config.get("open_scenario_id", "development-open-v1")])
    open_heuristics = [row for row in episodes if row["scenario_id"] in open_ids and row["method"] in ("local_frontier", "shared_frontier")]
    gates["heuristics_progress_on_open"] = bool(open_heuristics) and all(row["physical_coverage"] > row["initial_physical_coverage"] for row in open_heuristics)
    gates["matrix_complete"] = len(episodes) == len(methods) * len(scenarios) * len(seeds)
    gates["fixed_domain_consistent"] = all(len({(row["domain_sha256"], row["domain_cell_count"])
                                               for row in episodes if row["scenario_id"] == scenario["scenario_id"]}) == 1
                                            for scenario in scenarios)
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {**resources, "wall_seconds": time.perf_counter() - started,
                          "cpu_core_hours": (time.process_time() - cpu_started) / 3600},
            "summary": {"development_matrix_executed": True, "episodes": len(episodes), "groups": groups,
                        "heuristic_gate": "corridor_complete and every open-map local/shared frontier episode improves over initial physical coverage",
                        "primary_metric": "physical coverage on fixed reachable-free domain",
                        "auc_rule": "mean of coverage at time0 through horizon; carry final coverage after actual termination",
                        "primitive_step_unit": "one joint environment transition; decisions count two robot actions",
                        "scenario_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        "policy_modes": manifest["policy_modes"], "information_conditions": manifest["information_conditions"],
                        "replication_interpretation": "action-seed repeats of fixed maps; greedy controls are deterministic; no independent trained models or inferential performance claim",
                        "training": False, "known_pose_control": True},
            "required_artifacts": [manifest_path.name, trace_path.name]}
