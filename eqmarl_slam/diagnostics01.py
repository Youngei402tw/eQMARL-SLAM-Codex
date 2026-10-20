"""Deterministic Stage 01 numerical diagnostics with append-only raw records.

This module makes no training or quantum-performance claim. Its independent
finite differences evaluate the NumPy oracle rather than the Torch simulator.
"""

from __future__ import annotations

from contextlib import contextmanager
import resource
import sys
import time

import numpy as np


def run_diagnostics(config, emit, progress=None):
    """Return gates/resources and emit every measured fixture as a JSON object.

    The caller owns source publication, run identity, budget accounting, and
    durable archival. Optional config keys are fixture_seed, total_qubits,
    layers, finite_difference_steps, throughput_repetitions, and wall_seconds.
    Analytic CPU float64 and all finite-difference coordinates are mandatory.
    """
    import pennylane as qml
    import torch
    from .oracle import oracle_expectations, oracle_state
    from .quantum import QuantumCircuit

    torch.set_num_threads(1)
    started = time.perf_counter()
    cpu_started = time.process_time()
    rng = np.random.default_rng(int(config.get("fixture_seed", 1701)))
    qubit_sizes = config.get("total_qubits", [2, 4, 8])
    layers = int(config.get("layers", 2))
    steps = [float(h) for h in config.get("finite_difference_steps", [1e-4, 1e-5])]
    repetitions = int(config.get("throughput_repetitions", 3))
    if not qubit_sizes or any(n not in (2, 4, 8) for n in qubit_sizes) or layers < 1:
        raise ValueError("Stage 01 diagnostics require 2/4/8 wires and positive layers")
    if len(steps) < 2 or any(h <= 0 for h in steps) or any(a <= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Finite differences require at least two decreasing positive step sizes")
    if repetitions < 1:
        raise ValueError("Throughput repetitions must be positive")
    wall_cap = min(3600.0, float(config.get("wall_seconds", 3600)))
    counters = {"circuit_evaluations": 0, "simulations": 0, "shots": 0,
                "backward_additional_circuit_evaluations": 0, "oracle_evaluations": 0}
    if progress is not None:
        progress["resources"] = counters
    gates = {"bell_correlations": True, "expectation_parity": True, "state_norm": True,
             "gradients": True, "finite_difference_convergence": True,
             "batch_values": True, "batch_gradients": True, "repeated_inputs": True,
             "local_marginal_zero": True, "joint_input_sensitive": True}
    records = []

    def check_budget():
        if time.perf_counter() - started >= wall_cap:
            raise TimeoutError("Stage 01 numerical diagnostic family exhausted its wall-time cap")

    def record(row):
        check_budget()
        records.append(row)
        emit(row)

    def tensor(value):
        return torch.tensor(value, dtype=torch.float64, requires_grad=True)

    def numpy(value):
        return value.detach().cpu().numpy()

    @contextmanager
    def tracked(circuit):
        check_budget()
        tracker = qml.Tracker(circuit.device)
        try:
            with tracker:
                yield tracker
        finally:
            counters["circuit_evaluations"] += int(tracker.totals.get("executions", 0))
            counters["simulations"] += int(tracker.totals.get("simulations", 0))
            counters["shots"] += int(tracker.totals.get("shots", 0))

    def gradients(circuit, features, theta, scales):
        with tracked(circuit) as tracker:
            values = circuit(features, theta, scales)
            before = int(tracker.totals.get("executions", 0))
            grads = torch.autograd.grad(values.sum(), (features, theta, scales))
            counters["backward_additional_circuit_evaluations"] += (
                int(tracker.totals.get("executions", 0)) - before
            )
        return numpy(values), [numpy(grad) for grad in grads]

    def oracle_value(arrays, **kwargs):
        check_budget()
        counters["oracle_evaluations"] += 1
        return float(oracle_expectations(*arrays, **kwargs)[0])

    # Source Bell preparation is checked directly before applying any rotations.
    bell_expected = {"phi+": [1, -1, 1], "phi-": [-1, 1, 1],
                     "psi+": [1, 1, -1], "psi-": [-1, -1, -1]}
    for bell, expected in bell_expected.items():
        circuit = QuantumCircuit(2, 1, 1, bell=bell,
                                 observables=("XX", "YY", "ZZ", "XI", "YI", "ZI", "IX", "IY", "IZ"))
        with tracked(circuit):
            actual = numpy(circuit(tensor(np.zeros((2, 1))), tensor(np.zeros((2, 2, 1, 3))),
                                   tensor(np.ones((2, 1, 1, 1)))))
        expected = np.r_[expected, np.zeros(6)]
        error = float(np.max(np.abs(actual - expected)))
        passed = error <= 1e-8
        gates["bell_correlations"] &= passed
        record({"kind": "bell_correlations", "bell": bell, "total_qubits": 2,
                "observables": list(circuit.words), "actual": actual.tolist(),
                "expected": expected.tolist(), "max_absolute_error": error, "passed": bool(passed)})

    for total in qubit_sizes:
        d = total // 2
        feature = rng.uniform(-1.2, 1.2, size=(2, d))
        theta = rng.uniform(-1.1, 1.1, size=(2, layers + 1, d, 3))
        scales = rng.uniform(0.4, 1.4, size=(2, layers, d, 1))
        arrays = [feature, theta, scales]
        # Every Bell preparation plus its matched no-input-entanglement case.
        for entanglement, bell in [(True, name) for name in bell_expected] + [(False, "phi+")]:
            circuit = QuantumCircuit(2, d, layers, entanglement=entanglement, bell=bell, observables="all")
            with tracked(circuit):
                actual = numpy(circuit(*(tensor(array) for array in arrays)))
            expected = oracle_expectations(*arrays, entanglement=entanglement, bell=bell, observables="all")
            state = oracle_state(*arrays, entanglement=entanglement, bell=bell)
            counters["oracle_evaluations"] += 2
            error = float(np.max(np.abs(actual - expected)))
            norm_error = float(abs(np.vdot(state, state).real - 1))
            passed = error <= 1e-8 and norm_error <= 1e-10
            gates["expectation_parity"] &= error <= 1e-8
            gates["state_norm"] &= norm_error <= 1e-10
            record({"kind": "oracle_parity", "total_qubits": total, "layers": layers,
                    "entanglement": entanglement, "bell": bell, "observables": list(circuit.words),
                    "features": feature.tolist(), "theta": theta.tolist(), "scales": scales.tolist(),
                    "actual": actual.tolist(), "expected": expected.tolist(),
                    "max_absolute_error": error, "state_norm_error": norm_error, "passed": bool(passed)})

        for entanglement in (True, False):
            circuit = QuantumCircuit(2, d, layers, entanglement=entanglement)
            _, analytic = gradients(circuit, *(tensor(array) for array in arrays))
            for family_index, family in enumerate(("input", "theta", "scale")):
                errors = []
                fd_arrays = []
                for h in steps:
                    finite_difference = np.empty_like(arrays[family_index])
                    for coordinate in np.ndindex(finite_difference.shape):
                        plus = [array.copy() for array in arrays]
                        minus = [array.copy() for array in arrays]
                        plus[family_index][coordinate] += h
                        minus[family_index][coordinate] -= h
                        finite_difference[coordinate] = (
                            oracle_value(plus, entanglement=entanglement)
                            - oracle_value(minus, entanglement=entanglement)
                        ) / (2 * h)
                    fd_arrays.append(finite_difference)
                    errors.append(float(np.max(np.abs(analytic[family_index] - finite_difference))))
                finite = bool(np.isfinite(analytic[family_index]).all())
                connected = float(np.linalg.norm(analytic[family_index])) > 1e-10
                passed = finite and connected and all(
                    np.allclose(analytic[family_index], fd, atol=1e-6, rtol=1e-3) for fd in fd_arrays
                )
                # Rounding dominates below this floor; a smaller step must not
                # otherwise increase the truncation error materially.
                converged = all(b <= max(1.2 * a, 2e-9) for a, b in zip(errors, errors[1:]))
                gates["gradients"] &= bool(passed)
                gates["finite_difference_convergence"] &= bool(converged)
                record({"kind": "gradient_finite_difference", "total_qubits": total,
                        "entanglement": entanglement, "bell": "phi+", "parameter_family": family,
                        "coordinate_count": int(analytic[family_index].size), "steps": steps,
                        "analytic": analytic[family_index].tolist(),
                        "finite_differences": [fd.tolist() for fd in fd_arrays],
                        "max_absolute_errors": errors, "finite": finite, "connected": connected,
                        "step_convergence": bool(converged), "passed": bool(passed and converged)})

        # Identical inputs exercise repeatability; the third sample differs in
        # every feature so joint input-sensitivity is measured independently.
        features = np.stack([feature, feature, feature + rng.uniform(0.2, 0.7, feature.shape)])
        circuit = QuantumCircuit(2, d, layers)
        batch_values, batch_grads = gradients(circuit, tensor(features), tensor(theta), tensor(scales))
        scalar_features = tensor(features)
        scalar_theta, scalar_scales = tensor(theta), tensor(scales)
        with tracked(circuit) as tracker:
            scalar_values = torch.stack([circuit(sample, scalar_theta, scalar_scales) for sample in scalar_features])
            before = int(tracker.totals.get("executions", 0))
            scalar_grads = torch.autograd.grad(scalar_values.sum(), (scalar_features, scalar_theta, scalar_scales))
            counters["backward_additional_circuit_evaluations"] += int(tracker.totals.get("executions", 0)) - before
        value_error = float(np.max(np.abs(batch_values - numpy(scalar_values))))
        gradient_errors = [float(np.max(np.abs(batch - numpy(scalar))))
                           for batch, scalar in zip(batch_grads, scalar_grads)]
        repeat_value_error = float(np.max(np.abs(batch_values[0] - batch_values[1])))
        repeat_gradient_error = float(np.max(np.abs(batch_grads[0][0] - batch_grads[0][1])))
        joint_variance = float(np.var(batch_values))
        gates["batch_values"] &= value_error <= 1e-8
        gates["batch_gradients"] &= max(gradient_errors) <= 1e-8
        gates["repeated_inputs"] &= max(repeat_value_error, repeat_gradient_error) <= 1e-8
        gates["joint_input_sensitive"] &= joint_variance > 1e-12
        record({"kind": "batching", "total_qubits": total, "batch_size": len(features),
                "features": features.tolist(), "values": batch_values.tolist(),
                "scalar_value_max_error": value_error, "scalar_gradient_max_errors": gradient_errors,
                "repeat_value_max_error": repeat_value_error, "repeat_input_gradient_max_error": repeat_gradient_error,
                "joint_value_variance": joint_variance, "passed": bool(value_error <= 1e-8 and max(gradient_errors) <= 1e-8
                and max(repeat_value_error, repeat_gradient_error) <= 1e-8 and joint_variance > 1e-12)})

        local_circuit = QuantumCircuit(2, d, layers, observables="local")
        partition_circuit = QuantumCircuit(2, d, layers, observables="partition")
        local_values, local_grads = gradients(local_circuit, tensor(features), tensor(theta), tensor(scales))
        partition_values, partition_grads = gradients(partition_circuit, tensor(features), tensor(theta), tensor(scales))
        local_error = float(np.max(np.abs(local_values)))
        partition_error = float(np.max(np.abs(partition_values)))
        local_gradient_error = max(float(np.max(np.abs(gradient))) for gradient in local_grads + partition_grads)
        marginal_passed = max(local_error, partition_error, local_gradient_error) <= 1e-8
        gates["local_marginal_zero"] &= marginal_passed
        record({"kind": "bell_local_marginals", "total_qubits": total, "bell": "phi+",
                "local_Z_max_absolute": local_error, "partition_parity_max_absolute": partition_error,
                "local_gradient_max_absolute": local_gradient_error,
                "assumptions": "fully paired Bell inputs; subsequent gates restricted to each partition",
                "passed": bool(marginal_passed)})

        for repetition in range(repetitions):
            inputs = [tensor(features), tensor(theta), tensor(scales)]
            with tracked(circuit) as tracker:
                forward_started = time.perf_counter()
                values = circuit(*inputs)
                forward_wall = time.perf_counter() - forward_started
                forward_executions = int(tracker.totals.get("executions", 0))
                backward_started = time.perf_counter()
                torch.autograd.grad(values.sum(), inputs)
                backward_wall = time.perf_counter() - backward_started
                backward_executions = int(tracker.totals.get("executions", 0)) - forward_executions
                counters["backward_additional_circuit_evaluations"] += backward_executions
            record({"kind": "throughput", "total_qubits": total, "batch_size": len(features),
                    "repetition": repetition, "forward_wall_seconds": forward_wall,
                    "backward_wall_seconds": backward_wall, "forward_circuit_evaluations": forward_executions,
                    "backward_additional_circuit_evaluations": backward_executions,
                    "examples_per_second_forward_backward": len(features) / (forward_wall + backward_wall),
                    "ideal_statevector_bytes": 16 * 2**total, "resource_report": circuit.resource_report()})

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    counters.update({"wall_seconds": time.perf_counter() - started,
                     "cpu_core_hours": (time.process_time() - cpu_started) / 3600,
                     "gpu_device_hours": 0, "peak_rss_bytes": int(peak_rss if sys.platform == "darwin" else peak_rss * 1024),
                     "physical_shots_sampled": False,
                     "shot_interpretation": "analytic exact-expectation simulator; no hardware shot-cost claim"})
    return {"passed": bool(all(gates.values())), "gates": {key: bool(value) for key, value in gates.items()},
            "resources": counters, "fixture_seed": int(config.get("fixture_seed", 1701)),
            "tolerances": {"expectation_atol": 1e-8, "state_norm_atol": 1e-10,
                           "gradient_atol": 1e-6, "gradient_rtol": 1e-3,
                           "finite_difference_rounding_floor": 2e-9},
            "limits": {"total_qubits": list(qubit_sizes), "layers": layers, "wall_seconds": wall_cap,
                       "gradient_scope": "all input, variational and encoding-scale coordinates; phi+ and matched no-entanglement",
                       "training": False, "legacy_TFQ_parity": "separate_reference_environment_probe"},
            "record_count": len(records)}
