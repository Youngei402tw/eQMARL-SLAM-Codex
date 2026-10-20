"""Reportable, fixed-size Stage 01 objective and serialization diagnostics.

The caller must publish and verify its source/config commit before invoking
this function. These synthetic checks establish software contracts, not
exploration, SLAM performance, or historical TensorFlow training reproduction.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch

from .checkpoint import checkpoint_sha256, load_checkpoint, save_checkpoint
from .classical import SharedGRUActor
from .contracts import LocalObservationAdapter, TinyContractEnv
from .learner import corrected_a2c_terms, faithful_maa2c_terms


class _RecurrentCheckpointFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = torch.nn.GRUCell(3, 4, dtype=torch.float64)
        self.head = torch.nn.Linear(4, 1, dtype=torch.float64)

    def forward(self, observations, hidden):
        hidden = self.cell(observations, hidden)
        return self.head(hidden), hidden


def _stochastic_step(model, optimizer, hidden):
    observations = torch.randn(2, 3, dtype=torch.float64) + random.random() + float(np.random.randn())
    targets = torch.randn(2, 1, dtype=torch.float64)
    optimizer.zero_grad()
    predictions, next_hidden = model(observations, hidden)
    loss = (predictions - targets).square().mean()
    loss.backward()
    optimizer.step()
    return {
        "observations": observations.detach().clone(),
        "targets": targets.detach().clone(),
        "predictions": predictions.detach().clone(),
        "next_hidden": next_hidden.detach().clone(),
        "loss": loss.detach().clone(),
    }


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _exact_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(_exact_equal(x, y) for x, y in zip(a, b))
    return a == b


def run_contract_diagnostics(config: dict, output_dir: str | Path, emit: Callable[[dict], None]) -> dict:
    """Write fixed raw checks and checkpoints; return gates and actual counters.

    ``emit`` receives JSON-serializable dictionaries for append-only raw metrics.
    ``config['seed']`` defaults to 1701. No new scientific choices are made from
    outcomes. The fixture performs three synthetic optimizer updates (one
    warmup, original continuation, restored continuation), and three line-world
    API steps, all distinguished from research training/environment samples.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 1701))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gates = {}

    def gate(name, passed, **evidence):
        gates[name] = bool(passed)
        emit({"kind": "contract_gate", "gate": name, "passed": bool(passed), **_plain(evidence)})

    # Independent scalar/array NumPy implementation, with no use of Torch's
    # loss, reductions or target calculations in expected values.
    probs_np = np.asarray([[[.15, .35, .5], [.8, .1, .1]], [[.4, .4, .2], [.2, .5, .3]], [[.7, .2, .1], [.1, .2, .7]]], dtype=np.float64)
    actions_np = np.asarray([[0, 2], [1, 0], [2, 1]], dtype=np.int64)
    rewards_np = np.asarray([[.4, -.2], [-1.2, .1], [2., .5]], dtype=np.float64)
    values_np = np.asarray([[.3], [-.2], [1.1]], dtype=np.float64)
    next_values_np = np.asarray([[.9], [.5], [1.7]], dtype=np.float64)
    dones_np = np.asarray([[False, False], [False, True], [False, False]])
    gamma, alpha = .83, .037
    team_reward = np.asarray([[sum(row)] for row in rewards_np])
    team_done = np.asarray([[any(row)] for row in dones_np], dtype=np.float64)
    bootstrap = (1 - team_done) * gamma * next_values_np
    target = team_reward + bootstrap
    advantage = target - values_np
    chosen_logs = np.asarray([[np.log(probs_np[t, j, actions_np[t, j]]) for j in range(2)] for t in range(3)])
    entropy = float(np.mean([-sum(p * np.log(p) for p in row[-1]) for row in probs_np]))
    absolute_error = np.abs(target - values_np)
    critic_loss = np.where(absolute_error <= 1, .5 * absolute_error ** 2, absolute_error - .5).sum()
    expected = {
        "team_reward": team_reward, "team_done": team_done, "bootstrap": bootstrap,
        "target": target, "advantage": advantage, "chosen_log_probs": chosen_logs,
        "entropy": entropy, "actor_loss": np.mean(-chosen_logs * advantage) + alpha * entropy,
        "critic_loss": critic_loss,
    }
    probs = torch.tensor(probs_np, requires_grad=True)
    actions = torch.tensor(actions_np)
    rewards = torch.tensor(rewards_np)
    values = torch.tensor(values_np, requires_grad=True)
    next_values = torch.tensor(next_values_np, requires_grad=True)
    dones = torch.tensor(dones_np)
    emit({"kind": "objective_fixture_inputs", "fixture_id": "three-transition-two-agent-float64-v1", "probabilities": probs_np.tolist(), "actions": actions_np.tolist(), "rewards": rewards_np.tolist(), "values": values_np.tolist(), "next_values": next_values_np.tolist(), "dones": dones_np.tolist(), "gamma": gamma, "alpha": alpha})
    actual = faithful_maa2c_terms(probs, actions, rewards, values, next_values, dones, gamma=gamma, alpha=alpha)
    errors = []
    for term, expected_value in expected.items():
        observed = actual[term].detach().numpy()
        error = float(np.max(np.abs(observed - expected_value)))
        errors.append(error)
        emit({"kind": "faithful_objective_term", "term": term, "actual": observed.tolist(), "expected": _plain(expected_value), "max_absolute_error": error, "atol": 1e-12})
    gate("faithful_objective_numpy_parity", max(errors) <= 1e-12, maximum_absolute_error=max(errors), terms=list(expected))
    actual["critic_loss"].backward()
    faithful_bootstrap_gradient = float(next_values.grad.norm())
    gate("faithful_bootstrap_gradient_retained", faithful_bootstrap_gradient > 0, next_value_gradient_norm=faithful_bootstrap_gradient)

    # Corrected terms explicitly detach both target and actor advantage.
    probs.grad = None
    values.grad = None
    next_values.grad = None
    corrected = corrected_a2c_terms(probs, actions, rewards, values, next_values, dones, gamma=gamma, alpha=alpha)
    corrected["actor_loss"].backward(retain_graph=True)
    actor_isolated = values.grad is None and next_values.grad is None
    actor_probability_gradient = float(probs.grad.norm())
    corrected["critic_loss"].backward()
    detached = not corrected["target"].requires_grad and not corrected["advantage"].requires_grad
    gate("corrected_gradient_boundary", detached and actor_isolated and next_values.grad is None and values.grad is not None and actor_probability_gradient > 0, target_requires_grad=corrected["target"].requires_grad, advantage_requires_grad=corrected["advantage"].requires_grad, actor_loss_updates_critic_values=not actor_isolated, actor_probability_gradient_norm=actor_probability_gradient, critic_value_gradient_norm=float(values.grad.norm()), next_value_gradient=None if next_values.grad is None else _plain(next_values.grad))
    emit({"kind": "corrected_objective_terms", "terms": _plain(corrected), "disclosed_changes": ["detached target", "detached actor advantage", "all-agent entropy", "negative entropy coefficient"]})
    durations = torch.tensor([[3.], [2.], [1.]], dtype=torch.float64)
    terminal = torch.zeros_like(dones)
    truncation = torch.zeros_like(dones)
    truncation[0, 0] = True
    external = corrected_a2c_terms(probs, actions, rewards, values, next_values, terminal, truncated=truncation, durations=durations, gamma=.8)
    finite = corrected_a2c_terms(probs, actions, rewards, values, next_values, terminal, truncated=truncation, durations=durations, gamma=.8, finite_horizon=True)
    expected_bootstrap = .8 ** durations * next_values.detach()
    gate("duration_and_truncation_bootstrap", torch.equal(external["bootstrap"], expected_bootstrap) and finite["bootstrap"][0].item() == 0 and torch.equal(finite["bootstrap"][1:], external["bootstrap"][1:]), durations=_plain(durations), external_bootstrap=_plain(external["bootstrap"]), finite_horizon_bootstrap=_plain(finite["bootstrap"]))

    # One genuine checkpoint round trip of a synthetic shared-agent recurrent
    # network, including initialized Adam moments and stochastic draws from all
    # three RNG families. These are not policy-learning training updates.
    model = _RecurrentCheckpointFixture()
    optimizer = torch.optim.Adam(model.parameters(), lr=.012)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
    hidden = torch.zeros(2, 4, dtype=torch.float64)
    warmup = _stochastic_step(model, optimizer, hidden)
    hidden = warmup["next_hidden"]
    scheduler.step()
    initial_hash = save_checkpoint(output / "initial.pt", model, optimizer, recurrent_state=hidden, progress={"synthetic_updates": 1}, normalization={"mean": torch.tensor(.17), "count": 7}, scheduler=scheduler, environment_state={"fixture": "checkpoint-only", "steps": 0})
    original = _stochastic_step(model, optimizer, hidden)
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()
    continued_hash = save_checkpoint(output / "continued.pt", model, optimizer, recurrent_state=original["next_hidden"], progress={"synthetic_updates": 2}, normalization={"mean": torch.tensor(.17), "count": 7}, scheduler=scheduler, environment_state={"fixture": "checkpoint-only", "steps": 0})
    restored_model = _RecurrentCheckpointFixture()
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=.5)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=3)
    restored = load_checkpoint(output / "initial.pt", restored_model, restored_optimizer, expected_sha256=initial_hash, scheduler=restored_scheduler)
    continued = _stochastic_step(restored_model, restored_optimizer, restored["recurrent_state"])
    update_equal = _exact_equal(expected_model, restored_model.state_dict()) and _exact_equal(expected_optimizer, restored_optimizer.state_dict())
    outputs_equal = _exact_equal(original, continued)
    state_equal = restored["progress"] == {"synthetic_updates": 1} and restored["normalization"]["count"] == 7 and restored["environment_state"] == {"fixture": "checkpoint-only", "steps": 0} and _exact_equal(restored_scheduler.state_dict(), scheduler.state_dict())
    parameter_error = max(float((expected_model[name] - value).abs().max()) for name, value in restored_model.state_dict().items())
    gate("checkpoint_complete_next_update", update_equal and outputs_equal and state_equal, maximum_parameter_error=parameter_error, predictions_and_all_rng_draws_exact=outputs_equal, model_and_optimizer_exact=update_equal, state_metadata_exact=state_equal, original=_plain(original), restored=_plain(continued), checkpoints={"initial.pt": initial_hash, "continued.pt": continued_hash})

    # Gymnasium API fixture: both true termination and external cutoff happen
    # explicitly. Its known positions are not labeled as SLAM estimates.
    env = TinyContractEnv()
    obs, info = env.reset(seed=seed)
    api_valid = env.observation_space.contains(obs) and info["task_kind"] == "synthetic_known_state_contract_fixture"
    for index in range(2):
        obs, reward, terminated, truncated, info = env.step(np.array([2, 2]))
        api_valid = api_valid and env.observation_space.contains(obs) and isinstance(reward, float)
        emit({"kind": "synthetic_env_transition", "scenario": "true_terminal", "step": index + 1, "observation": obs.tolist(), "reward": reward, "terminated": terminated, "truncated": truncated})
        if index == 0:
            api_valid = api_valid and not terminated and not truncated
    api_valid = api_valid and terminated and not truncated
    limited = TinyContractEnv(max_steps=1)
    limited.reset(seed=seed)
    obs, reward, terminated, truncated, info = limited.step(np.array([1, 1]))
    emit({"kind": "synthetic_env_transition", "scenario": "external_cutoff", "step": 1, "observation": obs.tolist(), "reward": reward, "terminated": terminated, "truncated": truncated})
    gate("gymnasium_terminal_truncation_contract", api_valid and not terminated and truncated and limited.observation_space.contains(obs))

    adapter = LocalObservationAdapter(2)
    beliefs = {0: np.array([.1, .2]), 1: np.array([.3, .4])}
    own_before = adapter.from_agent(0, beliefs)
    beliefs[1] = np.array([1e6, -1e6])
    own_after = adapter.from_agent(0, beliefs)
    gate("observation_private_information_isolation", np.array_equal(own_before, own_after), own_input_before=own_before.tolist(), own_input_after=own_after.tolist(), teammate_private_belief_after=beliefs[1].tolist())

    actor = SharedGRUActor(3, 5, 3).double()
    observations = torch.tensor([[[.1, .2, 1.], [.8, -.5, 1.]]], dtype=torch.float64)
    hidden = actor.initial_state((1,), 2)
    hidden[0, 0] = .5
    hidden[0, 1] = -.5
    mask = torch.tensor([[[True, False, True], [False, True, False]]])
    distribution, updated = actor(observations, hidden, mask)
    altered_hidden = hidden.clone()
    altered_hidden[0, 1] += 10
    _, altered_updated = actor(observations, altered_hidden, mask)
    _, reset_updated = actor(observations, hidden, mask, torch.tensor([[True, False]]))
    zeroed = hidden.clone()
    zeroed[0, 0] = 0
    _, explicitly_reset = actor(observations, zeroed, mask)
    actor_loss = -distribution.log_prob(torch.tensor([[0, 1]])).sum()
    actor_loss.backward()
    encoder_norm = float(actor.encoder[0].weight.grad.norm())
    masking_valid = distribution.probs[0, 0, 1].item() == 0 and distribution.probs[0, 1, 1].item() == 1
    memory_isolated = torch.equal(updated[0, 0], altered_updated[0, 0])
    reset_exact = torch.equal(reset_updated, explicitly_reset)
    gate("actor_masks_per_agent_memory_and_resets", masking_valid and memory_isolated and reset_exact and encoder_norm > 0, probabilities=_plain(distribution.probs), own_hidden_after=_plain(updated[0, 0]), own_hidden_after_teammate_change=_plain(altered_updated[0, 0]), selective_reset_matches_explicit_zero=reset_exact, encoder_gradient_norm=encoder_norm)

    artifacts = []
    for name in ["initial.pt", "continued.pt"]:
        path = output / name
        artifacts.append({"path": name, "size_bytes": path.stat().st_size, "sha256": checkpoint_sha256(path), "purpose": "Synthetic checkpoint exact-continuation gate", "retention": "permanent"})
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "resources": {
            "primitive_steps": 0,
            "contract_env_steps": 3,
            "synthetic_optimization_updates": 3,
            "actor_gradient_evaluations": 1,
            "objective_gradient_evaluations": 3,
            "circuit_evaluations": 0,
            "shots": 0,
        },
        "artifacts": artifacts,
        "parameter_counts": {
            "checkpoint_fixture": sum(p.numel() for p in model.parameters()),
            "shared_actor_fixture": sum(p.numel() for p in actor.parameters()),
        },
        "limitations": "Synthetic fixed-batch and API contracts only; no policy training, exploration, estimator or SLAM performance was measured.",
    }
