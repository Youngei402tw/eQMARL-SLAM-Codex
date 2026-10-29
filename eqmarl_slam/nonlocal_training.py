"""Equal-objective training for the registered decentralized policy families."""
from __future__ import annotations

import math

import torch

from .nonlocal_game import (
    classical_local_expected_win, quantum_expected_win,
    shared_randomness_expected_win,
)

METHODS = ("classical_local", "classical_shared", "quantum_separable", "quantum_entangled")


def initialize(method: str, seed: int) -> dict[str, torch.nn.Parameter]:
    generator = torch.Generator().manual_seed(seed)
    if method == "classical_shared":
        return {"strategy_logits": torch.nn.Parameter(torch.randn(16, generator=generator, dtype=torch.float64) * 0.1)}
    if method == "classical_local":
        return {"alice_logits": torch.nn.Parameter(torch.randn(2, generator=generator, dtype=torch.float64) * 0.5),
                "bob_logits": torch.nn.Parameter(torch.randn(2, generator=generator, dtype=torch.float64) * 0.5)}
    if method in ("quantum_separable", "quantum_entangled"):
        return {"alpha": torch.nn.Parameter(torch.rand(2, generator=generator, dtype=torch.float64) * 2 * math.pi - math.pi),
                "beta": torch.nn.Parameter(torch.rand(2, generator=generator, dtype=torch.float64) * 2 * math.pi - math.pi)}
    raise ValueError(f"Unknown method: {method}")


def expected_win(method: str, parameters: dict[str, torch.Tensor], *, noise: float = 0.0) -> torch.Tensor:
    if method == "classical_local":
        return classical_local_expected_win(parameters["alice_logits"], parameters["bob_logits"])
    if method == "classical_shared":
        return shared_randomness_expected_win(parameters["strategy_logits"])
    if method == "quantum_separable":
        return quantum_expected_win(parameters["alpha"], parameters["beta"], entangled=False, noise=noise)
    if method == "quantum_entangled":
        return quantum_expected_win(parameters["alpha"], parameters["beta"], entangled=True, noise=noise)
    raise ValueError(f"Unknown method: {method}")


def train(method: str, seed: int, learning_rate: float, updates: int, *, record_every: int = 25) -> dict:
    parameters = initialize(method, seed)
    optimizer = torch.optim.Adam(parameters.values(), lr=learning_rate)
    curve = []
    gradients_finite = True
    for update in range(1, updates + 1):
        optimizer.zero_grad()
        win = expected_win(method, parameters)
        (-win).backward()
        gradients_finite = gradients_finite and all(torch.isfinite(value.grad).all().item() for value in parameters.values())
        optimizer.step()
        if update == 1 or update % record_every == 0 or update == updates:
            curve.append({"update": update, "expected_win": float(expected_win(method, parameters).detach())})
    return {"method": method, "seed": seed, "learning_rate": learning_rate, "updates": updates,
            "final_expected_win": float(expected_win(method, parameters).detach()),
            "parameters": {name: value.detach().tolist() for name, value in parameters.items()},
            "curve": curve, "gradients_finite": gradients_finite}


def restore(method: str, payload: dict[str, list]) -> dict[str, torch.Tensor]:
    expected = initialize(method, 0)
    if set(payload) != set(expected):
        raise ValueError("Checkpoint parameter names do not match method")
    restored = {name: torch.tensor(payload[name], dtype=torch.float64) for name in payload}
    if any(restored[name].shape != expected[name].shape for name in restored):
        raise ValueError("Checkpoint parameter shape mismatch")
    return restored
