"""Certified XOR requests and a consumable-entanglement allocation MDP."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch

EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def generate_request_types(seed: int, count: int, split: str) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        predicate = rng.integers(0, 2, size=(3, 3)).tolist()
        weights = rng.integers(1, 4, size=(3, 3)).tolist()
        rows.append({"type_id": f"{split}-type-{index:03d}", "predicate": predicate, "weights": weights})
    return {"schema_version": "budgeted-request-types-v1", "split": split, "generator_seed": seed,
            "contexts_per_agent": 3, "binary_actions": True, "types": rows}


def classical_value(request: dict) -> tuple[float, dict]:
    predicate = np.asarray(request["predicate"], dtype=int)
    weights = np.asarray(request["weights"], dtype=float); weights /= weights.sum()
    best, strategy = -1.0, None
    for alice in itertools.product((0, 1), repeat=3):
        for bob in itertools.product((0, 1), repeat=3):
            value = sum(weights[x, y] * (((alice[x] ^ bob[y]) == predicate[x, y])) for x in range(3) for y in range(3))
            if value > best:
                best, strategy = float(value), {"alice": list(alice), "bob": list(bob)}
    return best, strategy


def bell_win(angles: torch.Tensor, request: dict) -> torch.Tensor:
    if angles.shape != (6,):
        raise ValueError("three local measurement angles per endpoint are required")
    predicate = torch.tensor(request["predicate"], dtype=angles.dtype, device=angles.device)
    weights = torch.tensor(request["weights"], dtype=angles.dtype, device=angles.device)
    weights = weights / weights.sum()
    total = torch.zeros((), dtype=angles.dtype, device=angles.device)
    for x in range(3):
        for y in range(3):
            sign = 1 - 2 * predicate[x, y]
            total = total + weights[x, y] * (1 + sign * torch.cos(angles[x] - angles[3 + y])) / 2
    return total


def optimize_bell(request: dict, seed: int, starts: int = 16, updates: int = 800, learning_rate: float = 0.05) -> dict:
    best = None
    for start in range(starts):
        generator = torch.Generator().manual_seed(seed + start)
        angles = torch.nn.Parameter(torch.rand(6, generator=generator, dtype=torch.float64) * 2 * math.pi - math.pi)
        optimizer = torch.optim.Adam([angles], lr=learning_rate)
        for _ in range(updates):
            optimizer.zero_grad(); loss = -bell_win(angles, request); loss.backward(); optimizer.step()
        value = float(bell_win(angles, request).detach())
        row = {"value": value, "angles": angles.detach().tolist(), "start": start}
        if best is None or value > best["value"]:
            best = row
    return best


def xor_sdp_upper(request: dict) -> dict:
    import cvxpy as cp
    predicate = np.asarray(request["predicate"], dtype=int)
    weights = np.asarray(request["weights"], dtype=float); weights /= weights.sum()
    coefficient = weights * (1 - 2 * predicate)
    gram = cp.Variable((6, 6), symmetric=True)
    objective = cp.Maximize(cp.sum(cp.multiply(coefficient, gram[:3, 3:])))
    problem = cp.Problem(objective, [gram >> 0, cp.diag(gram) == 1])
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SDP did not solve: {problem.status}")
    return {"win_upper": float((1 + problem.value) / 2), "bias_upper": float(problem.value),
            "status": problem.status, "solver": "CLARABEL"}


def certify_request(request: dict, seed: int) -> dict:
    classical, strategy = classical_value(request)
    bell = optimize_bell(request, seed)
    sdp = xor_sdp_upper(request)
    return {"type_id": request["type_id"], "classical_value": classical, "classical_strategy": strategy,
            "bell_value": bell["value"], "bell_angles": bell["angles"], "sdp_upper": sdp["win_upper"],
            "quantum_gap": bell["value"] - classical, "sdp_gap": sdp["win_upper"] - bell["value"],
            "sdp_status": sdp["status"], "sdp_solver": sdp["solver"]}


def valid_matchings(active_edges: list[int], budget: int = 2) -> list[tuple[int, ...]]:
    output = [()]
    for size in range(1, min(2, budget) + 1):
        for subset in itertools.combinations(active_edges, size):
            endpoints = [vertex for edge_index in subset for vertex in EDGES[edge_index]]
            if len(endpoints) == len(set(endpoints)):
                output.append(tuple(subset))
    return output


def generate_public_catalog(seed: int, split: str, type_ids: list[str], states: int = 12) -> dict:
    rng = np.random.default_rng(seed)
    catalog = []
    for state in range(states):
        count = int(rng.integers(3, 6))
        active = sorted(rng.choice(len(EDGES), size=count, replace=False).tolist())
        requests = []
        for edge_index in active:
            urgency = int(rng.choice([1, 2, 5], p=[0.55, 0.30, 0.15]))
            requests.append({"edge_index": edge_index, "agents": list(EDGES[edge_index]),
                             "type_id": str(rng.choice(type_ids)), "urgency": urgency})
        catalog.append({"state_id": state, "requests": requests})
    raw = rng.random((states, states))
    for state in range(states):
        raw[state, state] += 8.0
        raw[state, (state + 1) % states] += 2.0
    transition = (raw / raw.sum(axis=1, keepdims=True)).tolist()
    return {"schema_version": "budgeted-public-catalog-v1", "split": split, "generator_seed": seed,
            "agents": 4, "horizon": 10, "bell_pair_budget": 6, "states": catalog, "transition": transition}


@dataclass
class BudgetedAllocationMDP:
    catalog: dict
    certificates: dict[str, dict]

    def actions(self, state_id: int, budget: int) -> list[tuple[int, ...]]:
        active = [row["edge_index"] for row in self.catalog["states"][state_id]["requests"]]
        return valid_matchings(active, budget)

    def action_gain(self, state_id: int, action: tuple[int, ...]) -> float:
        requests = {row["edge_index"]: row for row in self.catalog["states"][state_id]["requests"]}
        return float(sum(requests[edge]["urgency"] * max(0.0, self.certificates[requests[edge]["type_id"]]["quantum_gap"]) for edge in action))

    def classical_value(self, state_id: int) -> float:
        return float(sum(row["urgency"] * self.certificates[row["type_id"]]["classical_value"] for row in self.catalog["states"][state_id]["requests"]))

    def oracle_table(self) -> np.ndarray:
        horizon, max_budget = self.catalog["horizon"], self.catalog["bell_pair_budget"]
        states = len(self.catalog["states"]); transition = np.asarray(self.catalog["transition"])
        value = np.zeros((horizon + 1, states, max_budget + 1))
        for remaining in range(1, horizon + 1):
            for state in range(states):
                for budget in range(max_budget + 1):
                    future_by_budget = transition[state] @ value[remaining - 1]
                    value[remaining, state, budget] = max(self.action_gain(state, action) + future_by_budget[budget - len(action)] for action in self.actions(state, budget))
        return value

    def oracle_action(self, table: np.ndarray, remaining: int, state: int, budget: int) -> tuple[int, ...]:
        transition = np.asarray(self.catalog["transition"])
        future = transition[state] @ table[remaining - 1]
        return max(self.actions(state, budget), key=lambda action: self.action_gain(state, action) + future[budget - len(action)])

    def sample_sequence(self, seed: int, initial_state: int | None = None) -> list[int]:
        rng = np.random.default_rng(seed); states = len(self.catalog["states"])
        state = int(rng.integers(states)) if initial_state is None else initial_state
        sequence = []
        for _ in range(self.catalog["horizon"]):
            sequence.append(state); state = int(rng.choice(states, p=self.catalog["transition"][state]))
        return sequence

    def evaluate(self, sequence: list[int], policy) -> dict:
        budget = self.catalog["bell_pair_budget"]; surplus = classical = 0.0; actions = []
        for step, state in enumerate(sequence):
            action = tuple(policy(self, state, budget, len(sequence) - step))
            if action not in self.actions(state, budget):
                raise ValueError("policy returned infeasible matching")
            gain = self.action_gain(state, action); surplus += gain; classical += self.classical_value(state)
            budget -= len(action); actions.append({"state": state, "matching": list(action), "gain": gain, "budget_after": budget})
        return {"classical_return": classical, "entanglement_surplus": surplus, "total_return": classical + surplus,
                "pairs_used": self.catalog["bell_pair_budget"] - budget, "pairs_unused": budget, "actions": actions}
