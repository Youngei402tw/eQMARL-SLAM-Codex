"""Controlled complementarity falsifiers and throughput preflight."""

import time

import numpy as np
import torch

from .belief_task import BeliefTask, coordination_score
from .critics import make_critic, tracked
from .mappo import train


def run_diagnostics(config, output, emit, progress=None):
    torch.set_num_threads(1)
    resources = {
        "primitive_steps": 0,
        "decisions": 0,
        "updates": 0,
        "circuit_evaluations": 0,
        "shots": 0,
    }
    if progress is not None:
        progress["resources"] = resources
    gates = {}

    tables = {}
    interactions = []
    for condition in config["conditions"]:
        alpha = condition["complementarity_alpha"]
        table = {
            f"{first}{second}": coordination_score([first, second], alpha)
            for first in [0, 1]
            for second in [0, 1]
        }
        interaction = table["11"] - table["10"] - table["01"] + table["00"]
        tables[condition["name"]] = table
        interactions.append(interaction)
        emit(
            {
                "kind": "counterfactual_reward_table",
                "condition": condition["name"],
                "complementarity_alpha": alpha,
                "table": table,
                "balanced_mean": float(np.mean(list(table.values()))),
                "mixed_interaction": interaction,
            }
        )
    gates["counterfactual_reward_scale_matched"] = all(
        np.isclose(np.mean(list(table.values())), 1.0, atol=1e-12) for table in tables.values()
    )
    gates["mixed_interaction_monotone"] = bool(
        np.isclose(interactions[0], 0.0, atol=1e-12)
        and interactions[0] < interactions[1] < interactions[2]
    )

    isolation_task = {
        **config["task_base"],
        "complementarity_alpha": config["conditions"][0]["complementarity_alpha"],
    }
    isolation_env = BeliefTask(isolation_task, config["development_scenarios"][0])
    before, _ = isolation_env.reset()
    isolation_env.sim._poses[1, :2] += 1.0
    isolation_env.sim._landmarks[0] += 1.0
    private_change, _ = isolation_env.observe()
    gates["actor_truth_isolation"] = bool(np.array_equal(before, private_change))
    isolation_env.close()

    trajectories = {}
    for condition in config["conditions"]:
        task = {
            **config["task_base"],
            "complementarity_alpha": condition["complementarity_alpha"],
        }
        rows = []
        for scenario in config["development_scenarios"]:
            env = BeliefTask(task, scenario)
            observation, _ = env.reset()
            actions_trace = []
            reward = 0.0
            done = False
            rng = np.random.default_rng(config["action_seed"])
            while not done:
                actions = env.heuristic("frontier", rng)
                actions_trace.append(actions.tolist())
                resources["decisions"] += 2
                _, _, _, done, info = env.step(actions)
                resources["primitive_steps"] += info["duration"]
                reward += info["undiscounted_reward"]
                gates["reward_components_sum"] = gates.get("reward_components_sum", True) and all(
                    np.isfinite(sum(parts.values())) for parts in info["reward_components"]
                )
            metrics = env.metrics()
            row = {
                "kind": "scripted_mechanism_episode",
                "condition": condition["name"],
                "scenario_id": scenario["scenario_id"],
                "actions": actions_trace,
                "return": reward,
                **metrics,
            }
            rows.append(row)
            emit(row)
            env.close()
        trajectories[condition["name"]] = rows

    reference = trajectories["additive"]
    gates["reward_only_intervention_preserves_dynamics"] = all(
        row["actions"] == reference[index]["actions"]
        and row["primitive_steps"] == reference[index]["primitive_steps"]
        and np.isclose(row["physical_coverage"], reference[index]["physical_coverage"])
        for condition in ["low", "high"]
        for index, row in enumerate(trajectories[condition])
    )
    gates["mechanism_activated_in_scripted_task"] = any(
        row["coordination_event_counts"][3] > 0 for row in trajectories["high"]
    )

    critic = make_critic("flexible_interaction", config["critic_seed"])
    observations = torch.tensor(
        np.random.default_rng(config["critic_seed"]).normal(size=(16, 2, 27)),
        dtype=torch.float64,
    )
    targets = torch.tensor(
        np.random.default_rng(config["critic_seed"] + 1).normal(size=16),
        dtype=torch.float64,
    )
    values = critic(observations)
    loss = (values-targets).square().mean()
    loss.backward()
    gradient_groups = {
        name: float(parameter.grad.norm())
        for name, parameter in critic.named_parameters()
        if parameter.grad is not None
    }
    symmetric_error = float((critic(observations)-critic(observations.flip(-2))).abs().max().detach())
    gates["flexible_interaction_gradients"] = bool(
        gradient_groups and all(np.isfinite(value) and value > 0 for value in gradient_groups.values())
    )
    gates["flexible_interaction_symmetric"] = symmetric_error <= 1e-12
    emit(
        {
            "kind": "flexible_interaction_contract",
            "parameter_count": sum(parameter.numel() for parameter in critic.parameters()),
            "gradient_norms": gradient_groups,
            "agent_swap_max_absolute_error": symmetric_error,
        }
    )

    smoke = dict(config["smoke_training"])
    smoke["task"] = {
        **config["task_base"],
        "complementarity_alpha": config["conditions"][-1]["complementarity_alpha"],
    }
    smoke["training_scenarios"] = config["development_scenarios"]
    smoke["validation_scenarios"] = []
    quantum = make_critic("eqmarl", smoke["critic_seed"])
    smoke_progress = {}
    started = time.perf_counter()
    result = None
    with tracked(quantum, resources):
        result = train(smoke, output, emit, critic=quantum, progress=smoke_progress)
    elapsed = time.perf_counter() - started
    for key in ["primitive_steps", "decisions", "updates"]:
        resources[key] += result["resources"][key]
    measured_rate = elapsed / result["resources"]["primitive_steps"]
    historical_rate = config["historical_stage08_seconds_per_primitive"]
    quantum_primitives = config["confirmatory_projection"]["quantum_models"] * config["confirmatory_projection"]["training_primitives_per_model"]
    classical_primitives = config["confirmatory_projection"]["classical_models"] * config["confirmatory_projection"]["training_primitives_per_model"]
    projected = config["confirmatory_projection"]["safety_factor"] * (
        measured_rate*quantum_primitives + historical_rate*classical_primitives
    )
    projected *= 1 + config["confirmatory_projection"]["evaluation_overhead_fraction"]
    gates["quantum_training_smoke"] = bool(result["passed"])
    gates["confirmatory_projection_within_stage_cap"] = projected < config["stage_wall_cap_seconds"]
    emit(
        {
            "kind": "throughput_projection",
            "smoke_wall_seconds": elapsed,
            "smoke_primitives": result["resources"]["primitive_steps"],
            "measured_quantum_seconds_per_primitive": measured_rate,
            "historical_all_method_seconds_per_primitive": historical_rate,
            "projected_confirmatory_wall_seconds": projected,
            "stage_cap_seconds": config["stage_wall_cap_seconds"],
            "projection": config["confirmatory_projection"],
        }
    )

    return {
        "passed": all(gates.values()),
        "gates": gates,
        "resources": resources,
        "summary": {
            "counterfactual_interactions": interactions,
            "scripted_episodes": sum(len(rows) for rows in trajectories.values()),
            "flexible_interaction_parameters": sum(parameter.numel() for parameter in critic.parameters()),
            "projected_confirmatory_wall_seconds": projected,
            "test_manifest_accessed": False,
            "scope": "Task/critic correctness and throughput only; no comparative policy claim.",
        },
    }
