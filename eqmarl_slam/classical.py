"""Torch-only actor and team-critic interfaces; no quantum imports."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical


class SharedGRUActor(nn.Module):
    """Shared parameters with a separate hidden-state row for each agent.

    Shapes are ``(*batch, agents, features)`` and ``(*batch, agents, hidden)``.
    ``reset_mask`` identifies rows starting a new episode; rollout cutoffs alone
    must not reset recurrent state. ``True`` in the action mask means feasible.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        if min(obs_dim, hidden_dim, n_actions) < 1:
            raise ValueError("Actor dimensions must be positive")
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.Tanh())
        self.recurrent = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)

    def initial_state(self, batch_shape: Sequence[int] = (), agents: int = 2, *, device=None, dtype=None):
        parameter = next(self.parameters())
        return torch.zeros(
            (*batch_shape, agents, self.hidden_dim),
            device=device if device is not None else parameter.device,
            dtype=dtype if dtype is not None else parameter.dtype,
        )

    def forward(self, observations, hidden, action_mask, reset_mask=None):
        if observations.ndim < 2 or observations.shape[-1] != self.obs_dim:
            raise ValueError("Actor observations must have batch/agent axes and the declared feature dimension")
        leading = observations.shape[:-1]
        if hidden.shape != (*leading, self.hidden_dim):
            raise ValueError("Recurrent state must have one row per observation/agent")
        if action_mask.shape != (*leading, self.n_actions) or action_mask.dtype != torch.bool:
            raise ValueError("action_mask must be boolean with one feasibility flag per action")
        if not action_mask.any(dim=-1).all():
            raise ValueError("Each agent must have at least one feasible action")
        if not torch.isfinite(observations).all() or not torch.isfinite(hidden).all():
            raise ValueError("Actor inputs and recurrent states must be finite")
        if reset_mask is not None:
            if reset_mask.shape != leading or reset_mask.dtype != torch.bool:
                raise ValueError("reset_mask must be boolean with one flag per agent")
            hidden = hidden.masked_fill(reset_mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(observations).reshape(-1, self.hidden_dim)
        next_hidden = self.recurrent(encoded, hidden.reshape(-1, self.hidden_dim)).reshape(
            *leading, self.hidden_dim
        )
        logits = self.policy_head(next_hidden).masked_fill(~action_mask, -torch.inf)
        return Categorical(logits=logits), next_hidden


class CentralValueCritic(nn.Module):
    """Explicit training-time joint features to one scalar team value.

    The caller owns the protocol's critic information permission; this module
    never feeds its joint input into the decentralized actor.
    """

    def __init__(self, n_agents: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()
        if min(n_agents, feature_dim, hidden_dim) < 1:
            raise ValueError("Critic dimensions must be positive")
        self.n_agents = n_agents
        self.feature_dim = feature_dim
        self.value = nn.Sequential(
            nn.Linear(n_agents * feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, training_features):
        if training_features.ndim < 2 or training_features.shape[-2:] != (self.n_agents, self.feature_dim):
            raise ValueError("Critic features must have the declared agent and feature axes")
        joint = training_features.flatten(start_dim=-2)
        return self.value(joint).squeeze(-1)
