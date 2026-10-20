"""Fixed-batch learner contracts, with separately named source and corrected paths.

The faithful arithmetic is adapted from ``news-vt/eqmarl`` at commit
6b1b2e3817f661da81aad5e7188a362380dbdd1d, ``eqmarl/algorithms/maa2c.py``
(CC BY 4.0; see THIRD_PARTY_NOTICES.md).  It deliberately retains the source's
one-step bootstrap, non-detached target, final-agent entropy, and positive
entropy-loss sign.  This is fixed-batch arithmetic parity, not an assertion
that the legacy TensorFlow runtime or historical training was reproduced.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _check_inputs(action_probs, actions, rewards, values, next_values, flags):
    if action_probs.ndim != 3:
        raise ValueError("action_probs must have shape (time, agents, actions)")
    shape = action_probs.shape[:2]
    if actions.shape != shape or rewards.shape != shape or flags.shape != shape:
        raise ValueError("actions, rewards and flags must have shape (time, agents)")
    if values.shape != (shape[0], 1) or next_values.shape != values.shape:
        raise ValueError("values and next_values must have shape (time, 1)")
    if not action_probs.is_floating_point():
        raise ValueError("action_probs must be floating point")
    if not torch.isfinite(action_probs).all() or torch.any(action_probs <= 0):
        raise ValueError("fixed-batch parity requires finite positive probabilities")
    if not torch.allclose(action_probs.sum(-1), torch.ones_like(action_probs[..., 0])):
        raise ValueError("action probabilities must sum to one")
    if actions.is_floating_point() or torch.any(actions < 0) or torch.any(actions >= action_probs.shape[-1]):
        raise ValueError("actions must be in-range integer indices")
    for name, value in [("rewards", rewards), ("values", values), ("next_values", next_values)]:
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if not torch.all((flags == 0) | (flags == 1)):
        raise ValueError("termination flags must be binary")


def faithful_maa2c_terms(
    action_probs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    *,
    gamma: float = 1.0,
    alpha: float = 0.001,
) -> dict[str, Tensor]:
    """Reproduce the selected upstream MAA2C update's pre-optimizer terms.

    Rewards are summed over agents; any agent done suppresses the team
    bootstrap.  ``dones`` follows the source contract, which does not store
    truncation in its transition.  Huber delta is 1 and its per-transition
    scalar losses are summed, matching Keras ``Huber(reduction=SUM)`` for a
    single team value.  Source advantages and next-state values stay attached.
    """
    _check_inputs(action_probs, actions, rewards, values, next_values, dones)
    team_reward = rewards.sum(-1, keepdim=True)
    team_done = (dones.sum(-1, keepdim=True) > 0).to(values.dtype)
    bootstrap = (1 - team_done) * gamma * next_values
    target = team_reward + bootstrap
    advantage = target - values
    log_probs = action_probs.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1).log()
    # In the source update, `action_probs` after the actor loop refers to the
    # final agent, rather than the stacked `agents_action_probs` tensor.
    entropy = -(action_probs[:, -1] * action_probs[:, -1].log()).sum(-1).mean()
    actor_loss = (-log_probs * advantage).mean() + alpha * entropy
    critic_loss = F.huber_loss(values, target, reduction="sum", delta=1.0)
    return {
        "team_reward": team_reward,
        "team_done": team_done,
        "bootstrap": bootstrap,
        "target": target,
        "advantage": advantage,
        "chosen_log_probs": log_probs,
        "entropy": entropy,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
    }


def corrected_a2c_terms(
    action_probs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    *,
    truncated: Tensor | None = None,
    durations: Tensor | None = None,
    finite_horizon: bool = False,
    gamma: float = 1.0,
    alpha: float = 0.001,
) -> dict[str, Tensor]:
    """Explicit corrected one-step A2C diagnostic, separate from MAPPO.

    External truncation bootstraps. A true finite-horizon task boundary also
    suppresses the bootstrap; callers must include remaining budget in that
    task's actor input. Rewards must already contain the duration-discounted
    sum of primitive rewards for a macro action; only the bootstrap discount
    is raised to ``duration`` here. Entropy includes every agent and enters
    minimization with a negative sign. Targets and actor advantages detach.
    """
    _check_inputs(action_probs, actions, rewards, values, next_values, terminated)
    if truncated is None:
        truncated = torch.zeros_like(terminated)
    if truncated.shape != terminated.shape or not torch.all((truncated == 0) | (truncated == 1)):
        raise ValueError("truncated must have the same binary shape as terminated")
    if durations is None:
        durations = torch.ones_like(values)
    if durations.shape != values.shape or torch.any(durations < 1) or not torch.isfinite(durations).all():
        raise ValueError("durations must be finite positive primitive-step counts (time, 1)")
    if torch.any(durations != durations.round()):
        raise ValueError("durations must count whole primitive steps")
    if not 0 <= gamma <= 1 or alpha < 0:
        raise ValueError("gamma must lie in [0,1] and alpha must be nonnegative")
    team_reward = rewards.sum(-1, keepdim=True)
    team_done = terminated.bool().any(-1, keepdim=True)
    if finite_horizon:
        team_done = team_done | truncated.bool().any(-1, keepdim=True)
    team_done = team_done.to(values.dtype)
    bootstrap = (1 - team_done) * (gamma ** durations) * next_values.detach()
    target = (team_reward + bootstrap).detach()
    advantage = (target - values).detach()
    log_probs = action_probs.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1).log()
    entropy = -(action_probs * action_probs.log()).sum(-1).mean()
    actor_loss = (-log_probs * advantage).mean() - alpha * entropy
    critic_loss = F.huber_loss(values, target, reduction="sum", delta=1.0)
    return {
        "team_reward": team_reward,
        "team_done": team_done,
        "bootstrap": bootstrap,
        "target": target,
        "advantage": advantage,
        "chosen_log_probs": log_probs,
        "entropy": entropy,
        "actor_loss": actor_loss,
        "critic_loss": critic_loss,
    }
