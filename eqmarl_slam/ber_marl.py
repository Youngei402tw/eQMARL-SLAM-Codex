"""Budgeted Entanglement Routing with masked shared-action DQN."""
from __future__ import annotations

from collections import deque
import copy
import random

import numpy as np
import torch

from .budgeted_entanglement import EDGES, BudgetedAllocationMDP, valid_matchings


GLOBAL_ACTIONS = tuple(valid_matchings(list(range(len(EDGES))), budget=2))
ACTION_INDEX = {action: index for index, action in enumerate(GLOBAL_ACTIONS)}


class RoutingQ(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(14, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, len(GLOBAL_ACTIONS)),
        )

    def forward(self, features):
        return self.layers(features)


def features(mdp: BudgetedAllocationMDP, state: int, budget: int, remaining: int) -> np.ndarray:
    gains = np.zeros(6, dtype=np.float32)
    active = np.zeros(6, dtype=np.float32)
    for request in mdp.catalog["states"][state]["requests"]:
        edge = request["edge_index"]
        gains[edge] = mdp.action_gain(state, (edge,)) * 10.0
        active[edge] = 1.0
    return np.concatenate((gains, active, np.asarray([budget / 6.0, remaining / 10.0], dtype=np.float32)))


def feasible_indices(mdp: BudgetedAllocationMDP, state: int, budget: int) -> list[int]:
    return [ACTION_INDEX[action] for action in mdp.actions(state, budget)]


def dqn_action(model: RoutingQ, mdp: BudgetedAllocationMDP, state: int, budget: int, remaining: int) -> tuple[int, ...]:
    indices = feasible_indices(mdp, state, budget)
    with torch.no_grad():
        q = model(torch.from_numpy(features(mdp, state, budget, remaining)))
    return GLOBAL_ACTIONS[max(indices, key=lambda index: float(q[index]))]


def train_dqn(mdp: BudgetedAllocationMDP | list[BudgetedAllocationMDP], seed: int, *, episodes: int, hidden: int,
              learning_rate: float, batch_size: int = 64, target_interval: int = 200) -> tuple[RoutingQ, dict]:
    """Train from public transitions only; rewards are exact expected request surplus."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    environments = mdp if isinstance(mdp, list) else [mdp]
    model = RoutingQ(hidden).to(dtype=torch.float32)
    target = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay = deque(maxlen=20000)
    rng = np.random.default_rng(seed + 100000)
    losses, returns = [], []
    updates = decisions = 0
    for episode in range(episodes):
        environment_index = episode % len(environments)
        mdp = environments[environment_index]
        transition = np.asarray(mdp.catalog["transition"])
        horizon = mdp.catalog["horizon"]
        state = int(rng.integers(len(mdp.catalog["states"])))
        budget = mdp.catalog["bell_pair_budget"]
        episode_return = 0.0
        epsilon = max(0.03, 1.0 - episode / max(1, episodes * 0.65))
        for step in range(horizon):
            remaining = horizon - step
            valid = feasible_indices(mdp, state, budget)
            if rng.random() < epsilon:
                action_index = int(rng.choice(valid))
            else:
                with torch.no_grad():
                    q = model(torch.from_numpy(features(mdp, state, budget, remaining)))
                action_index = max(valid, key=lambda index: float(q[index]))
            action = GLOBAL_ACTIONS[action_index]
            reward = mdp.action_gain(state, action)
            next_state = int(rng.choice(len(transition), p=transition[state]))
            next_budget = budget - len(action)
            done = step == horizon - 1
            replay.append((features(mdp, state, budget, remaining), action_index, reward * 10.0,
                           features(mdp, next_state, next_budget, remaining - 1), next_state, next_budget, done,
                           environment_index))
            state, budget = next_state, next_budget
            episode_return += reward; decisions += 1
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                obs = torch.from_numpy(np.stack([row[0] for row in batch]))
                actions = torch.tensor([row[1] for row in batch], dtype=torch.long)
                rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)
                next_obs = torch.from_numpy(np.stack([row[3] for row in batch]))
                predicted = model(obs).gather(1, actions[:, None]).squeeze(1)
                targets = []
                with torch.no_grad():
                    next_q = target(next_obs)
                    for index, row in enumerate(batch):
                        if row[6]:
                            targets.append(rewards[index])
                        else:
                            valid_next = feasible_indices(environments[row[7]], row[4], row[5])
                            targets.append(rewards[index] + next_q[index, valid_next].max())
                loss = torch.nn.functional.smooth_l1_loss(predicted, torch.stack(targets))
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step(); losses.append(float(loss.detach())); updates += 1
                if updates % target_interval == 0:
                    target.load_state_dict(model.state_dict())
        returns.append(episode_return)
    return model, {"episodes": episodes, "decisions": decisions, "updates": updates,
                   "final_100_return": float(np.mean(returns[-100:])),
                   "final_100_loss": float(np.mean(losses[-100:]))}


def evaluate_policy(model: RoutingQ, mdp: BudgetedAllocationMDP, sequence_seeds: list[int]) -> dict:
    surplus = []
    for seed in sequence_seeds:
        sequence = mdp.sample_sequence(seed)
        result = mdp.evaluate(sequence, lambda env, state, budget, remaining: dqn_action(model, env, state, budget, remaining))
        surplus.append(result["entanglement_surplus"])
    return {"mean_surplus": float(np.mean(surplus)), "surplus": surplus}


def greedy_action(mdp, state, budget, remaining):
    return max(mdp.actions(state, budget), key=lambda action: mdp.action_gain(state, action))


def static_action(priorities: list[int], mdp, state, budget, remaining):
    valid = mdp.actions(state, budget)
    return min(valid, key=lambda action: tuple(priorities.index(edge) if edge in priorities else 99 for edge in action) + (-len(action),))
