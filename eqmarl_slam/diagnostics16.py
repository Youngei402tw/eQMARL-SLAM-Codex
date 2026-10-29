"""Stage 16 correctness and throughput gates for learned nonlocal coordination."""
from __future__ import annotations

import math
import time

import numpy as np
import pennylane as qml
import torch

from .nonlocal_game import (
    CLASSICAL_CEILING, CONTEXTS, TSIRELSON_WIN, InformationContract,
    bell_optimal_angles, deterministic_strategy_scores, quantum_expected_win,
    sampled_win_rate,
)


def _qml_context(alpha: float, beta: float, entangled: bool) -> float:
    device = qml.device("default.qubit", wires=2)

    @qml.qnode(device)
    def circuit():
        if entangled:
            qml.Hadamard(0)
            qml.CNOT([0, 1])
        qml.RY(alpha, 0)
        qml.RY(beta, 1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return float(circuit())


def run_diagnostics(config, output, emit, progress=None):
    started = time.perf_counter()
    strategies = deterministic_strategy_scores()
    classical_max = max(row["expected_win"] for row in strategies)
    alpha, beta = bell_optimal_angles()
    expected = float(quantum_expected_win(alpha, beta, entangled=True))
    separable_candidate = float(quantum_expected_win(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]), entangled=False))

    qml_correlations = []
    analytic_correlations = []
    signs = []
    for x, y in CONTEXTS:
        qml_value = _qml_context(float(alpha[x]), float(beta[y]), True)
        analytic = math.cos(float(alpha[x] - beta[y]))
        qml_correlations.append(qml_value)
        analytic_correlations.append(analytic)
        signs.append(-1.0 if x & y else 1.0)
        emit({"kind": "circuit_fixture", "x": x, "y": y, "qml_correlation": qml_value, "analytic_correlation": analytic})

    train_alpha = torch.tensor([0.31, -0.72], dtype=torch.float64, requires_grad=True)
    train_beta = torch.tensor([1.17, 0.42], dtype=torch.float64, requires_grad=True)
    value = quantum_expected_win(train_alpha, train_beta, entangled=True)
    gradients = torch.autograd.grad(value, (train_alpha, train_beta))
    epsilon = 1e-6
    finite = []
    for group, base in ((0, train_alpha.detach()), (1, train_beta.detach())):
        values = []
        for index in range(2):
            plus_a, minus_a = train_alpha.detach().clone(), train_alpha.detach().clone()
            plus_b, minus_b = train_beta.detach().clone(), train_beta.detach().clone()
            (plus_a if group == 0 else plus_b)[index] += epsilon
            (minus_a if group == 0 else minus_b)[index] -= epsilon
            values.append(float((quantum_expected_win(plus_a, plus_b, entangled=True) - quantum_expected_win(minus_a, minus_b, entangled=True)) / (2 * epsilon)))
        finite.append(values)
    gradient_error = max(abs(float(gradients[group][index]) - finite[group][index]) for group in range(2) for index in range(2))

    decisions = int(config["sampling_decisions"])
    sampled, wins = sampled_win_rate(expected, decisions, int(config["seed_streams"]["quantum_shots"]))
    sampling_z = abs(sampled - expected) / math.sqrt(expected * (1 - expected) / decisions)
    emit({"kind": "sample_fixture", "decisions": decisions, "wins": wins, "sampled_win": sampled, "exact_win": expected})

    # A separable state with local measurements is a local stochastic policy. The
    # exhaustive deterministic vertices certify its convex ceiling; random global
    # optimization checks the implemented parameterization reaches but does not cross it.
    optimized = []
    for seed in range(32):
        generator = torch.Generator().manual_seed(16000 + seed)
        a = torch.nn.Parameter(torch.rand(2, generator=generator, dtype=torch.float64) * 2 * math.pi - math.pi)
        b = torch.nn.Parameter(torch.rand(2, generator=generator, dtype=torch.float64) * 2 * math.pi - math.pi)
        optimizer = torch.optim.Adam([a, b], lr=0.08)
        for _ in range(400):
            optimizer.zero_grad(); loss = -quantum_expected_win(a, b, entangled=False); loss.backward(); optimizer.step()
        optimized.append(float(quantum_expected_win(a, b, entangled=False)))

    contract = InformationContract()
    gates = {
        "classical_bound_exact": classical_max == CLASSICAL_CEILING,
        "bell_optimum_exact": abs(expected - TSIRELSON_WIN) <= 1e-12,
        "pennylane_matches_independent_formula": max(abs(a - b) for a, b in zip(qml_correlations, analytic_correlations)) <= 1e-12,
        "separable_reaches_classical_ceiling": abs(separable_candidate - CLASSICAL_CEILING) <= 1e-12 and max(optimized) <= CLASSICAL_CEILING + 1e-9,
        "gradient_finite_difference": gradient_error <= 1e-8,
        "sampled_matches_exact": sampling_z <= 4.0,
        "private_information_isolated": contract.alice_fields == ("x",) and contract.bob_fields == ("y",) and contract.communication_after_input_bits == 0,
    }
    resources = {"primitive_steps": decisions, "decisions": decisions, "updates": 32 * 400,
                 "circuit_evaluations": 4, "shots": 0}
    return {"passed": all(gates.values()), "gates": gates, "resources": resources,
            "summary": {"classical_strategy_count": len(strategies), "classical_max": classical_max,
                        "strategies_at_max": sum(row["expected_win"] == classical_max for row in strategies),
                        "tsirelson_win": TSIRELSON_WIN, "bell_policy_win": expected,
                        "qml_correlations": qml_correlations, "separable_candidate": separable_candidate,
                        "separable_optimization_max": max(optimized), "gradient_max_abs_error": gradient_error,
                        "sampled_win": sampled, "sampling_z": sampling_z,
                        "information_contract": contract.__dict__, "diagnostic_wall_seconds": time.perf_counter() - started}}
