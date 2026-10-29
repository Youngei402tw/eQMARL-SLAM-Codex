"""Two-agent CHSH cooperative contextual bandit and policy contracts."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch

CONTEXTS = ((0, 0), (0, 1), (1, 0), (1, 1))
CLASSICAL_CEILING = 0.75
TSIRELSON_WIN = math.cos(math.pi / 8) ** 2


def reward(x: int, y: int, a: int, b: int) -> int:
    """Shared CHSH reward; private inputs are used only by the environment."""
    return int((a ^ b) == (x & y))


def deterministic_strategy_scores() -> list[dict]:
    """Enumerate all local deterministic policies, including shared-randomness vertices."""
    rows = []
    for alice in itertools.product((0, 1), repeat=2):
        for bob in itertools.product((0, 1), repeat=2):
            wins = [reward(x, y, alice[x], bob[y]) for x, y in CONTEXTS]
            rows.append({"alice": list(alice), "bob": list(bob), "wins": wins, "expected_win": sum(wins) / 4})
    return rows


def _sign(x: int, y: int) -> float:
    return -1.0 if x & y else 1.0


def quantum_correlations(alpha: torch.Tensor, beta: torch.Tensor, *, entangled: bool, noise: float = 0.0) -> torch.Tensor:
    """ZZ correlations after local RY measurements on Bell or product |00> state."""
    if alpha.shape != (2,) or beta.shape != (2,):
        raise ValueError("alpha and beta must each contain one angle per private input")
    if not 0 <= noise <= 1:
        raise ValueError("noise must be a local depolarizing probability in [0,1]")
    values = []
    for x, y in CONTEXTS:
        correlation = torch.cos(alpha[x] - beta[y]) if entangled else torch.cos(alpha[x]) * torch.cos(beta[y])
        values.append(correlation * (1 - noise) ** 2)
    return torch.stack(values)


def quantum_expected_win(alpha: torch.Tensor, beta: torch.Tensor, *, entangled: bool, noise: float = 0.0) -> torch.Tensor:
    signs = torch.tensor([_sign(x, y) for x, y in CONTEXTS], dtype=alpha.dtype, device=alpha.device)
    return ((1 + signs * quantum_correlations(alpha, beta, entangled=entangled, noise=noise)) / 2).mean()


def classical_local_expected_win(alice_logits: torch.Tensor, bob_logits: torch.Tensor) -> torch.Tensor:
    """Expected reward of factorized Bernoulli local policies."""
    if alice_logits.shape != (2,) or bob_logits.shape != (2,):
        raise ValueError("each local policy needs two logits")
    pa, pb = torch.sigmoid(alice_logits), torch.sigmoid(bob_logits)
    values = []
    for x, y in CONTEXTS:
        parity_equal = pa[x] * pb[y] + (1 - pa[x]) * (1 - pb[y])
        values.append(1 - parity_equal if x & y else parity_equal)
    return torch.stack(values).mean()


def shared_randomness_expected_win(strategy_logits: torch.Tensor) -> torch.Tensor:
    """Convex mixture over all 16 deterministic local strategy pairs."""
    scores = torch.tensor([row["expected_win"] for row in deterministic_strategy_scores()], dtype=strategy_logits.dtype, device=strategy_logits.device)
    if strategy_logits.shape != scores.shape:
        raise ValueError("shared-randomness policy needs one logit per deterministic strategy")
    return torch.dot(torch.softmax(strategy_logits, dim=0), scores)


def bell_optimal_angles(dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([0.0, math.pi / 2], dtype=dtype), torch.tensor([math.pi / 4, -math.pi / 4], dtype=dtype)


def sampled_win_rate(probability: float, decisions: int, seed: int) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    wins = int(rng.binomial(decisions, probability))
    return wins / decisions, wins


@dataclass(frozen=True)
class InformationContract:
    alice_fields: tuple[str, ...] = ("x",)
    bob_fields: tuple[str, ...] = ("y",)
    communication_after_input_bits: int = 0
    shared_randomness_independent_of_inputs: bool = True
    reward_privileged_inputs: tuple[str, ...] = ("x", "y", "a", "b")
