"""Isolated, pinned Cirq/TFQ fixture adapter; run with the legacy Python 3.9.

This module is intentionally independent of Torch/PennyLane. It imports the
unchanged source extracts through a private package namespace so that upstream's
large ``__init__`` does not run. It changes no upstream gate or layer code.
One-wire partitions are explicitly unsupported by the upstream self-CZ ring.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import types


SOURCE_COMMIT = "6b1b2e3817f661da81aad5e7188a362380dbdd1d"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_original(source_directory):
    """Load unchanged modules while bypassing upstream's unrelated imports."""
    source_directory = Path(source_directory).resolve()
    package_name = "_eqmarl_pinned_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(source_directory)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    loaded = {}
    for name in ("tools", "ops", "circuits", "layers"):
        path = source_directory / (name + ".py")
        spec = importlib.util.spec_from_file_location(package_name + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def evaluate_case(case, original, *, float64_atol, tfq_atol, check_gradients=False,
                  counters=None):
    import cirq
    import numpy as np
    import tensorflow as tf

    feature = np.asarray(case.get("features", case.get("feature")), dtype=np.float64)
    theta = np.asarray(case["theta"], dtype=np.float64)
    scales = np.asarray(case["scales"], dtype=np.float64)
    if feature.ndim != 2:
        raise ValueError("Each fixture must have unbatched features (parts,d)")
    parts, d = feature.shape
    layers = theta.shape[1] - 1
    if theta.shape != (parts, layers + 1, d, 3) or scales.shape != (parts, layers, d, 1):
        raise ValueError("Inconsistent fixture theta/scales dimensions")
    if d < 2:
        return {"id": case.get("id"), "status": "unsupported_upstream",
                "total_qubits": parts * d,
                "reason": "Pinned circular_entangling_layer attempts self-CZ for d=1; unchanged source is not executable for this diagnostic."}
    expected = case.get("expected", case.get("numpy_expectation"))
    if expected is None:
        raise ValueError("Every executed fixture requires a committed independent NumPy expected global parity")
    expected = float(np.asarray(expected).reshape(-1)[0])
    squash = case.get("squash", "atan")
    if squash not in ("atan", "arctan", "linear", "tanh"):
        raise ValueError("Unsupported encoding squash")
    entanglement = bool(case.get("entanglement", True))
    bell = case.get("bell", "phi+")
    qubits = cirq.LineQubit.range(parts * d)
    circuit, symbols = original["circuits"].generate_partite_variational_encoding_circuit(
        qubits, parts, d, layers, decompose=True,
        input_entanglement=entanglement, input_entanglement_type=bell)
    enc = scales * feature[:, None, :, None]
    if squash in ("atan", "arctan"):
        enc = np.arctan(enc)
    elif squash == "tanh":
        enc = np.tanh(enc)
    substitutions = {str(s): float(v) for array_s, array_v in zip(symbols, (theta, enc))
                     for s, v in zip(array_s.flat, array_v.flat)}
    resolved = cirq.resolve_parameters(circuit, substitutions)
    if counters is not None:
        counters["cirq_forward_circuit_evaluations"] += 1
    state = cirq.Simulator(dtype=np.complex128).simulate(resolved, qubit_order=qubits).final_state_vector
    observable = cirq.PauliString({q: cirq.Z for q in qubits})
    cirq_value = float(np.real(observable.expectation_from_state_vector(
        state, {q: i for i, q in enumerate(qubits)})))
    norm_error = float(abs(np.vdot(state, state).real - 1.0))

    # Use the original layer's explicit symbol sorting, parameter tiling,
    # weighted input encoding, and TFQ ControlledPQC implementation unchanged.
    qlayer = original["layers"].HybridPartiteVariationalEncodingPQC(
        qubits, parts, d, layers, [observable], squash_activation=squash,
        input_entanglement=entanglement, input_entanglement_type=bell)
    qlayer.w_var.assign(theta.astype(np.float32))
    qlayer.w_enc.assign(scales.astype(np.float32))
    inputs = tf.Variable(feature[None, ..., None].astype(np.float32))
    if counters is not None:
        counters["tfq_forward_circuit_evaluations"] += 1
    if check_gradients:
        with tf.GradientTape() as tape:
            output = qlayer(inputs)
            scalar = tf.reduce_sum(output)
        grads = tape.gradient(scalar, [inputs, qlayer.w_var, qlayer.w_enc])
    else:
        output = qlayer(inputs)
        grads = []
    tfq_value = float(output.numpy().reshape(-1)[0])
    gradients = {}
    for label, gradient in zip(("features", "theta", "scales"), grads):
        gradients[label] = None if gradient is None else gradient.numpy().tolist()
    finite_gradients = (all(gradient is not None and np.isfinite(gradient.numpy()).all()
                            for gradient in grads) if check_gradients else None)
    cirq_error = abs(cirq_value - expected)
    tfq_error = abs(tfq_value - expected)
    passed = bool(cirq_error <= float64_atol and norm_error <= 1e-10
                  and tfq_error <= tfq_atol and (finite_gradients is not False))
    return {"id": case.get("id"), "status": "completed", "passed": passed,
            "total_qubits": parts * d, "partitions": parts, "qubits_per_partition": d,
            "layers": layers, "entanglement": entanglement, "bell": bell,
            "squash": squash, "expected_numpy_global_parity": expected,
            "cirq_complex128_global_parity": cirq_value, "cirq_absolute_error": cirq_error,
            "cirq_state_norm_error": norm_error,
            "tfq_float32_global_parity": tfq_value, "tfq_absolute_error": tfq_error,
            "tfq_gradients": gradients if check_gradients else None,
            "tfq_gradients_tested": bool(check_gradients),
            "tfq_gradients_finite": finite_gradients,
            "float64_atol": float64_atol, "tfq_atol": tfq_atol,
            "source_gate_count": len(list(resolved.all_operations())),
            "source_moment_depth": len(resolved),
            "resources": {"cirq_forward_circuit_evaluations": 1,
                          "tfq_forward_circuit_evaluations": 1,
                          "tfq_backward_circuit_evaluations": None if check_gradients else 0,
                          "shots": 0,
                          "physically_sampled": False},
            "resource_note": ("Uninstrumented TFQ native adjoint gradient; not used in the registered forward-only comparison."
                              if check_gradients else
                              "Exactly one Cirq complex128 forward simulation and one original analytic TFQ forward; no backward requested or shots sampled.")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-dir", default=str(Path(__file__).resolve().parents[1] / "third_party" / "eqmarl"))
    parser.add_argument("--float64-atol", type=float, default=1e-8)
    parser.add_argument("--tfq-atol", type=float, default=2e-5,
                        help="Predeclared float32 TFQ tolerance; never replaces float64 gate")
    parser.add_argument("--gradients", action="store_true",
                        help="Optional exploratory TFQ gradients (native backward counts unknown); disabled in registered forward-only comparison")
    args = parser.parse_args(argv)
    if args.float64_atol <= 0 or args.float64_atol > 1e-8 or args.tfq_atol <= 0:
        parser.error("Float64 tolerance must be positive and at most 1e-8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/eqmarl-reference-matplotlib")
    started = time.perf_counter()
    cpu_started = time.process_time()
    fixture_path = Path(args.fixtures)
    fixtures = json.loads(fixture_path.read_text())
    if isinstance(fixtures, dict):
        fixtures = fixtures["fixtures"]
    if not isinstance(fixtures, list) or not fixtures:
        parser.error("Fixtures must be a nonempty list or an object with fixtures list")
    source_dir = Path(args.source_dir)
    counters = {"cirq_forward_circuit_evaluations": 0,
                "tfq_forward_circuit_evaluations": 0,
                "tfq_backward_circuit_evaluations": None if args.gradients else 0,
                "shots": 0, "physically_sampled": False}
    result = {"schema_version": "0.1.0", "upstream_source_commit": SOURCE_COMMIT,
              "fixtures_sha256": _sha256(fixture_path), "python": sys.version,
              "platform": platform.platform(), "packages": {},
              "source_sha256": {name: _sha256(source_dir / name) for name in
                                ("tools.py", "ops.py", "circuits.py", "layers.py")},
              "adaptations": ["Private import namespace bypasses upstream __init__ only.",
                              "Cirq LineQubit labels replace grid labels with identical explicit wire ordering.",
                              "Original one-wire self-CZ case is recorded unsupported; no source gate changed."],
              "claims": "Circuit/encoding forward parity only; no historical training reproduction.",
              "cases": []}
    try:
        original = load_original(source_dir)
        result["packages"] = {name: importlib.metadata.version(name) for name in
                              ("tensorflow", "tensorflow-quantum", "cirq-core", "numpy", "sympy")}
        for fixture in fixtures:
            try:
                row = evaluate_case(fixture, original,
                                    float64_atol=args.float64_atol, tfq_atol=args.tfq_atol,
                                    check_gradients=args.gradients, counters=counters)
            except Exception as exc:
                row = {"id": fixture.get("id"), "status": "failed", "passed": False,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()}
            result["cases"].append(row)
        completed = [row for row in result["cases"] if row["status"] == "completed"]
        result["passed"] = bool(len(completed) == len(fixtures)
                                and all(row.get("passed", False) for row in completed))
        result["gates"] = {
            "cirq_float64": bool(completed and all(row["cirq_absolute_error"] <= args.float64_atol for row in completed)),
            "tfq_float32": bool(completed and all(row["tfq_absolute_error"] <= args.tfq_atol for row in completed)),
            "state_norm": bool(completed and all(row["cirq_state_norm_error"] <= 1e-10 for row in completed)),
            "all_registered_fixtures_executed": len(completed) == len(fixtures),
        }
    except Exception as exc:
        result.update({"passed": False, "failure": {"error_type": type(exc).__name__,
                       "error": str(exc), "traceback": traceback.format_exc()}})
    result["wall_seconds"] = time.perf_counter() - started
    counters.update({"circuit_evaluations": counters["cirq_forward_circuit_evaluations"] + counters["tfq_forward_circuit_evaluations"],
                     "backward_additional_circuit_evaluations": None if args.gradients else 0,
                     "wall_seconds": result["wall_seconds"],
                     "cpu_core_hours": (time.process_time() - cpu_started) / 3600,
                     "gpu_device_hours": 0,
                     "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    result["resources"] = counters
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "cases": len(result["cases"]),
                      "output": str(output), "wall_seconds": result["wall_seconds"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
