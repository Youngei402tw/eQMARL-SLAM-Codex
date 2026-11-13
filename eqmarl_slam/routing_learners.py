"""Equal-information DQN and PPO allocators for robust entanglement routing."""
from __future__ import annotations

from collections import deque
import copy
import random
import time

import numpy as np
import torch

from .robust_routing import MAX_EDGES, RobustRoutingMDP

GLOBAL_DIM = 8


def action_features(mdp: RobustRoutingMDP, state: int, budget: int, remaining: int,
                    action: tuple[int, ...], mode: str = "full") -> tuple[np.ndarray, np.ndarray]:
    gains = mdp.edge_gains(state).astype(np.float32)
    active = np.zeros(len(mdp.edges), dtype=np.float32)
    active[[row["edge_index"] for row in mdp.catalog["states"][state]["requests"]]] = 1.0
    if mode == "binary_gain":
        observed = (gains > 0).astype(np.float32) * 0.02
    elif mode == "shuffled_gain":
        observed = gains[np.random.default_rng(state + 991).permutation(len(gains))]
    else:
        observed = gains
    selected = np.zeros(len(mdp.edges), dtype=np.float32)
    if action:
        selected[list(action)] = 1.0
    edge = np.zeros((MAX_EDGES, 3), dtype=np.float32)
    edge[:len(gains), 0] = observed * 10.0
    edge[:len(gains), 1] = active
    edge[:len(gains), 2] = selected
    selected_gain = float(observed[list(action)].sum()) if action else 0.0
    positive = observed[active > 0]
    global_features = np.asarray([
        0.0 if mode == "no_budget" else budget / max(1, mdp.catalog["bell_pair_budget"]),
        0.0 if mode == "no_horizon" else remaining / max(1, mdp.catalog["horizon"]),
        mdp.catalog["agents"] / 8.0, len(action) / 2.0, selected_gain * 10.0,
        (float(positive.max()) if len(positive) else 0.0) * 10.0,
        (float(positive.mean()) if len(positive) else 0.0) * 10.0,
        float(active.sum()) / MAX_EDGES,
    ], dtype=np.float32)
    return edge, global_features


class ActionScore(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.edge = torch.nn.Sequential(torch.nn.Linear(3, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.head = torch.nn.Sequential(torch.nn.Linear(hidden + GLOBAL_DIM, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, edge: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        if edge.ndim == 2:
            edge, global_features = edge.unsqueeze(0), global_features.unsqueeze(0)
        mask = edge[..., 1:2]
        pooled = (self.edge(edge) * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.head(torch.cat((pooled, global_features), dim=-1)).squeeze(-1)


class StateValue(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = ActionScore(hidden)

    def forward(self, edge, global_features):
        return self.score(edge, global_features)


def _tensor_features(rows):
    return (torch.from_numpy(np.stack([r[0] for r in rows])), torch.from_numpy(np.stack([r[1] for r in rows])))


def choose_q(model: ActionScore, mdp: RobustRoutingMDP, state: int, budget: int, remaining: int,
             mode: str = "full") -> tuple[int, ...]:
    actions = mdp.actions(state, budget)
    rows = [action_features(mdp, state, budget, remaining, action, mode) for action in actions]
    edge, glob = _tensor_features(rows)
    with torch.no_grad():
        values = model(edge, glob)
    return actions[int(values.argmax())]


def train_q(maps: list[RobustRoutingMDP], seed: int, episodes: int, hidden: int, learning_rate: float,
            *, double: bool, feature_mode: str = "full", batch_size: int = 64, target_interval: int = 200):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    online = ActionScore(hidden); target = copy.deepcopy(online)
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    replay = deque(maxlen=20000); rng = np.random.default_rng(seed + 100003)
    updates = decisions = 0; returns = []; losses = []
    started = time.perf_counter()
    for episode in range(episodes):
        mdp_index = episode % len(maps); mdp = maps[mdp_index]
        transition = np.asarray(mdp.catalog["transition"]); state = int(rng.integers(len(transition)))
        budget = int(mdp.catalog["bell_pair_budget"]); episode_return = 0.0
        epsilon = max(0.03, 1.0 - episode / max(1, episodes * 0.65))
        for step in range(int(mdp.catalog["horizon"])):
            remaining = int(mdp.catalog["horizon"]) - step; actions = mdp.actions(state, budget)
            if rng.random() < epsilon:
                action = actions[int(rng.integers(len(actions)))]
            else:
                action = choose_q(online, mdp, state, budget, remaining, feature_mode)
            reward = mdp.action_gain(state, action); next_state = int(rng.choice(len(transition), p=transition[state]))
            next_budget = budget - len(action); done = step + 1 == mdp.catalog["horizon"]
            replay.append((mdp_index, state, budget, remaining, action, reward * 10.0, next_state, next_budget, done))
            state, budget = next_state, next_budget; episode_return += reward; decisions += 1
            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)
                rows = [action_features(maps[r[0]], r[1], r[2], r[3], r[4], feature_mode) for r in batch]
                edge, glob = _tensor_features(rows); predicted = online(edge, glob)
                targets = []
                with torch.no_grad():
                    for row in batch:
                        if row[8]:
                            targets.append(row[5]); continue
                        next_mdp = maps[row[0]]; next_remaining = row[3] - 1
                        candidates = next_mdp.actions(row[6], row[7])
                        next_rows = [action_features(next_mdp, row[6], row[7], next_remaining, a, feature_mode) for a in candidates]
                        ne, ng = _tensor_features(next_rows)
                        if double:
                            chosen = int(online(ne, ng).argmax())
                            next_value = target(ne, ng)[chosen]
                        else:
                            next_value = target(ne, ng).max()
                        targets.append(row[5] + float(next_value))
                loss = torch.nn.functional.smooth_l1_loss(predicted, torch.tensor(targets, dtype=torch.float32))
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0); optimizer.step()
                updates += 1; losses.append(float(loss.detach()))
                if updates % target_interval == 0:
                    target.load_state_dict(online.state_dict())
        returns.append(episode_return)
    return online, {"episodes": episodes, "decisions": decisions, "updates": updates,
                    "final_100_return": float(np.mean(returns[-100:])), "final_100_loss": float(np.mean(losses[-100:])),
                    "wall_seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in online.parameters())}


class FeedForwardActorCritic(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.policy = ActionScore(hidden); self.value = StateValue(hidden)

    def distribution_value(self, mdp, state, budget, remaining, mode="full"):
        actions = mdp.actions(state, budget)
        rows = [action_features(mdp, state, budget, remaining, a, mode) for a in actions]
        edge, glob = _tensor_features(rows); logits = self.policy(edge, glob)
        state_edge, state_global = action_features(mdp, state, budget, remaining, (), mode)
        value = self.value(torch.from_numpy(state_edge), torch.from_numpy(state_global)).squeeze()
        return actions, torch.distributions.Categorical(logits=logits), value


class RecurrentActorCritic(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.state_edge = torch.nn.Sequential(torch.nn.Linear(2, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, hidden), torch.nn.Tanh())
        self.gru = torch.nn.GRUCell(hidden + GLOBAL_DIM, hidden)
        self.action = torch.nn.Sequential(torch.nn.Linear(hidden + 2, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        self.value = torch.nn.Linear(hidden, 1)

    def step(self, mdp, state, budget, remaining, hidden_state, mode="full"):
        empty_edge, empty_global = action_features(mdp, state, budget, remaining, (), mode)
        edge = torch.from_numpy(empty_edge[:, :2]); mask = edge[:, 1:2]
        pooled = (self.state_edge(edge) * mask).sum(0) / mask.sum().clamp_min(1.0)
        hidden_state = self.gru(torch.cat((pooled, torch.from_numpy(empty_global))), hidden_state)
        actions = mdp.actions(state, budget); action_rows = []
        gains = mdp.edge_gains(state)
        for action in actions:
            action_rows.append([len(action) / 2.0, float(gains[list(action)].sum()) * 10.0 if action else 0.0])
        a = torch.tensor(action_rows, dtype=torch.float32)
        logits = self.action(torch.cat((hidden_state.expand(len(actions), -1), a), dim=1)).squeeze(1)
        return actions, torch.distributions.Categorical(logits=logits), self.value(hidden_state).squeeze(), hidden_state


def train_ppo(maps: list[RobustRoutingMDP], seed: int, episodes: int, hidden: int, learning_rate: float,
              *, recurrent: bool, feature_mode: str = "full", rollout_episodes: int = 16, epochs: int = 4):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = RecurrentActorCritic(hidden) if recurrent else FeedForwardActorCritic(hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate); rng = np.random.default_rng(seed + 200003)
    decisions = updates = 0; returns_log = []; losses = []; started = time.perf_counter()
    for batch_start in range(0, episodes, rollout_episodes):
        trajectories = []
        for episode in range(batch_start, min(episodes, batch_start + rollout_episodes)):
            mdp_index = episode % len(maps); mdp = maps[mdp_index]; trans = np.asarray(mdp.catalog["transition"])
            state = int(rng.integers(len(trans))); budget = int(mdp.catalog["bell_pair_budget"])
            hidden_state = torch.zeros(hidden); trajectory = []; episode_return = 0.0
            for step in range(int(mdp.catalog["horizon"])):
                remaining = int(mdp.catalog["horizon"]) - step
                with torch.no_grad():
                    if recurrent:
                        actions, dist, value, hidden_state = model.step(mdp, state, budget, remaining, hidden_state, feature_mode)
                    else:
                        actions, dist, value = model.distribution_value(mdp, state, budget, remaining, feature_mode)
                    choice = dist.sample(); logp = dist.log_prob(choice)
                action = actions[int(choice)]; reward = mdp.action_gain(state, action)
                trajectory.append({"mdp": mdp_index, "state": state, "budget": budget, "remaining": remaining,
                                   "action_index": int(choice), "old_logp": float(logp), "reward": reward * 10.0})
                state = int(rng.choice(len(trans), p=trans[state])); budget -= len(action); episode_return += reward; decisions += 1
            running = 0.0
            for row in reversed(trajectory):
                running = row["reward"] + running; row["return"] = running
            trajectories.append(trajectory); returns_log.append(episode_return)
        all_returns = torch.tensor([row["return"] for traj in trajectories for row in traj])
        norm_mean, norm_std = all_returns.mean(), all_returns.std().clamp_min(1e-6)
        for _ in range(epochs):
            policy_terms, value_terms, entropy_terms = [], [], []
            for trajectory in trajectories:
                hidden_state = torch.zeros(hidden)
                for row in trajectory:
                    mdp = maps[row["mdp"]]
                    if recurrent:
                        actions, dist, value, hidden_state = model.step(mdp, row["state"], row["budget"], row["remaining"], hidden_state, feature_mode)
                    else:
                        actions, dist, value = model.distribution_value(mdp, row["state"], row["budget"], row["remaining"], feature_mode)
                    choice = torch.tensor(row["action_index"]); logp = dist.log_prob(choice)
                    advantage = (torch.tensor(row["return"]) - value).detach()
                    ratio = torch.exp(logp - row["old_logp"])
                    policy_terms.append(-torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage))
                    target = (torch.tensor(row["return"]) - norm_mean) / norm_std
                    value_terms.append((value - target).square()); entropy_terms.append(dist.entropy())
            loss = torch.stack(policy_terms).mean() + 0.5 * torch.stack(value_terms).mean() - 0.01 * torch.stack(entropy_terms).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach())); updates += 1
    return model, {"episodes": episodes, "decisions": decisions, "updates": updates,
                   "final_100_return": float(np.mean(returns_log[-100:])), "final_100_loss": float(np.mean(losses[-100:])),
                   "wall_seconds": time.perf_counter() - started, "parameters": sum(p.numel() for p in model.parameters())}


def choose_ppo(model, mdp, state, budget, remaining, *, recurrent=False, hidden_state=None, mode="full"):
    with torch.no_grad():
        if recurrent:
            actions, dist, _, hidden_state = model.step(mdp, state, budget, remaining, hidden_state, mode)
        else:
            actions, dist, _ = model.distribution_value(mdp, state, budget, remaining, mode)
        return actions[int(dist.logits.argmax())], hidden_state


def evaluate_model(model, method: str, mdp: RobustRoutingMDP, seeds: list[int], mode="full") -> list[float]:
    values = []
    for seed in seeds:
        sequence = mdp.sample_sequence(seed)
        if method in ("dqn", "double_dqn"):
            policy = lambda env, state, budget, remaining: choose_q(model, env, state, budget, remaining, mode)
        elif method == "ppo":
            policy = lambda env, state, budget, remaining: choose_ppo(model, env, state, budget, remaining, mode=mode)[0]
        else:
            hidden = torch.zeros(model.gru.hidden_size)
            def policy(env, state, budget, remaining):
                nonlocal hidden
                action, hidden = choose_ppo(model, env, state, budget, remaining, recurrent=True, hidden_state=hidden, mode=mode)
                return action
        values.append(mdp.evaluate(sequence, policy, seed=seed + 800000)["surplus"])
    return values
