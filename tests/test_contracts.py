"""Information-isolation and multi-agent boundary tests."""

import subprocess
import sys

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from eqmarl_slam.classical import CentralValueCritic, SharedGRUActor
from eqmarl_slam.contracts import LocalObservationAdapter, TinyContractEnv


def test_local_actor_isolation_and_field_rejection():
    adapter = LocalObservationAdapter(2)
    beliefs = {0: np.array([0.1, 0.2]), 1: np.array([0.3, 0.4])}
    own = adapter.from_agent(0, beliefs)
    beliefs[1] = np.array([1e6, -1e6])
    np.testing.assert_array_equal(own, adapter.from_agent(0, beliefs))
    with pytest.raises(ValueError, match="declared local"):
        adapter.from_fields({"own_belief": [0.1, 0.2], "truth": [0.4]})
    with pytest.raises(ValueError, match="cannot receive"):
        adapter.build([0.1, 0.2], [0.3])
    own[:] = 0
    np.testing.assert_array_equal(beliefs[0], [0.1, 0.2])


def test_message_boundary_is_explicit():
    adapter = LocalObservationAdapter(2, 1, condition="message-limited")
    with pytest.raises(ValueError, match="delivered-message"):
        adapter.build([0, 1])
    np.testing.assert_array_equal(adapter.build([0, 1], [2]), [0, 1, 2, 1])
    with pytest.raises(ValueError, match="remaining_budget"):
        adapter.build([0, 1], [2], remaining_budget=2)


def test_gymnasium_contract_and_boundary_semantics():
    env = TinyContractEnv()
    check_env(env, skip_render_check=True)
    obs, info = env.reset(seed=123)
    assert env.observation_space.contains(obs)
    assert info["task_kind"] == "synthetic_known_state_contract_fixture"
    obs, reward, terminal, truncated, info = env.step(np.array([2, 2]))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and not terminal and not truncated
    _, reward, terminal, truncated, _ = env.step(np.array([2, 2]))
    assert reward == 1 and terminal and not truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.array([2, 2]))
    limited = TinyContractEnv(max_steps=1)
    limited.reset()
    obs, _, terminal, truncated, _ = limited.step(np.array([1, 1]))
    assert not terminal and truncated and np.all(obs[:, -1] == 0)


def test_actor_separate_agent_memory_masking_and_selective_resets():
    torch.manual_seed(123)
    actor = SharedGRUActor(3, 5, 3).double()
    observations = torch.tensor([[[0.1, 0.2, 1], [0.8, -0.5, 1]]], dtype=torch.float64)
    hidden = actor.initial_state((1,), 2)
    hidden[0, 0] = 0.5
    hidden[0, 1] = -0.5
    mask = torch.tensor([[[True, False, True], [False, True, False]]])
    dist, updated = actor(observations, hidden, mask)
    assert updated.shape == hidden.shape
    assert dist.probs[0, 0, 1] == 0 and dist.probs[0, 1, 1] == 1
    changed = hidden.clone()
    changed[0, 1] += 10
    _, other_updated = actor(observations, changed, mask)
    torch.testing.assert_close(updated[0, 0], other_updated[0, 0], rtol=0, atol=0)
    _, reset_updated = actor(observations, hidden, mask, torch.tensor([[True, False]]))
    zeroed = hidden.clone()
    zeroed[0, 0] = 0
    _, explicit_updated = actor(observations, zeroed, mask)
    torch.testing.assert_close(reset_updated, explicit_updated, rtol=0, atol=0)
    with pytest.raises(ValueError, match="feasible"):
        actor(observations, hidden, torch.zeros_like(mask))
    loss = -dist.log_prob(torch.tensor([[0, 1]])).sum()
    loss.backward()
    assert actor.encoder[0].weight.grad.norm() > 0


def test_team_critic_contract_and_classical_import_isolation():
    critic = CentralValueCritic(2, 3)
    assert critic(torch.zeros(4, 2, 3)).shape == (4,)
    with pytest.raises(ValueError, match="axes"):
        critic(torch.zeros(4, 3, 2))
    result = subprocess.run(
        [sys.executable, "-c", "import eqmarl_slam.classical, eqmarl_slam.contracts, sys; "
         "assert not {'pennylane','cirq','tensorflow'} & set(sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
