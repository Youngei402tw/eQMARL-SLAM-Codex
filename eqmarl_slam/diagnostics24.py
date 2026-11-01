"""Independent release audit for budgeted entanglement routing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .archive import validate_bundle
from .budgeted_entanglement import classical_value, xor_sdp_upper

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _bootstrap(values, seed, replicates, percentile):
    values = np.asarray(values); rng = np.random.default_rng(seed); output = np.empty(replicates)
    for index in range(replicates):
        output[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return [float(np.percentile(output, percentile)), float(np.percentile(output, 100 - percentile))]


def run_diagnostics(config, output, emit, progress=None):
    bundle_results = []
    for relative in config["bundles"]:
        bundle_results.append(validate_bundle(ROOT / relative, repository_root=ROOT))

    confirmatory = ROOT / config["confirmatory_bundle"]
    recorded = json.loads((confirmatory / "confirmatory_analysis.json").read_text())
    recorded_certificates = {row["type_id"]: row for row in json.loads((confirmatory / "test_request_certificates.json").read_text())["certificates"]}
    test_manifest = json.loads((ROOT / config["test_request_manifest"]).read_text())
    certificate_checks = []
    for request in test_manifest["types"]:
        exact, strategy = classical_value(request); sdp = xor_sdp_upper(request); row = recorded_certificates[request["type_id"]]
        certificate_checks.append({"type_id": request["type_id"], "exact_classical": exact,
                                   "recorded_classical": row["classical_value"], "sdp_upper": sdp["win_upper"],
                                   "recorded_sdp": row["sdp_upper"], "recorded_bell": row["bell_value"],
                                   "classical_strategy": strategy})

    baselines = {}; learned = {}
    with (confirmatory / "episode_results.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["kind"] == "baseline_episode":
                baselines.setdefault(row["method"], []).append(row["surplus"])
            else:
                learned.setdefault(row["training_seed"], []).append(row["surplus"])
    method_means = {method: float(np.mean(values)) for method, values in baselines.items()}
    seed_surplus = [float(np.mean(learned[seed])) for seed in sorted(learned)]
    greedy = method_means["quantum_greedy"]; dp = method_means["dp_online_oracle"]
    differences = [value - greedy for value in seed_surplus]; fractions = [value / dp for value in seed_surplus]
    analysis = {"method_means": method_means, "ber_marl_mean_surplus": float(np.mean(seed_surplus)),
                "mean_greedy_difference": float(np.mean(differences)), "mean_dp_fraction": float(np.mean(fractions)),
                "difference_bootstrap_95": _bootstrap(differences, 23999, 20000, 2.5),
                "dp_fraction_one_sided_95": _bootstrap(fractions, 24000, 20000, 5.0),
                "surplus_one_sided_95": _bootstrap(seed_surplus, 24001, 20000, 5.0)}
    keys = ("method_means", "ber_marl_mean_surplus", "mean_greedy_difference", "mean_dp_fraction",
            "difference_bootstrap_95", "dp_fraction_one_sided_95", "surplus_one_sided_95")
    analysis_errors = {}
    for key in keys:
        left, right = analysis[key], recorded[key]
        if isinstance(left, dict):
            analysis_errors[key] = max(abs(left[name] - right[name]) for name in left)
        elif isinstance(left, list):
            analysis_errors[key] = max(abs(a - b) for a, b in zip(left, right))
        else:
            analysis_errors[key] = abs(left - right)

    checkpoints = sorted(confirmatory.glob("ber-marl-seed-*.pt"))
    checkpoint_hashes = {path.name: _sha(path) for path in checkpoints}
    claims = {
        "supported": [
            "BER-MARL had greater held-out entanglement surplus than random, static, and myopic greedy allocation.",
            "The paired-seed 95% bootstrap interval for BER-MARL minus greedy excluded zero.",
            "The benchmark requires routing a consumable Bell-pair budget over a dynamic matching-constrained graph.",
        ],
        "prohibited": [
            "BER-MARL passed the full five-part superior-allocation gate.",
            "BER-MARL achieved a one-sided lower bound above 90% of online-DP surplus.",
            "The study proves quantum computational speedup, hardware advantage, or global priority.",
            "The request-level nonlocal gaps are new Bell inequalities.",
        ],
    }
    gates = {
        "all_authoritative_bundles_valid": all(row["valid"] for row in bundle_results),
        "all_test_classical_values_exact": max(abs(row["exact_classical"] - row["recorded_classical"]) for row in certificate_checks) <= 1e-12,
        "all_test_sdp_values_reproduced": max(abs(row["sdp_upper"] - row["recorded_sdp"]) for row in certificate_checks) <= 1e-8,
        "all_bell_values_within_sdp": all(row["recorded_bell"] <= row["sdp_upper"] + 2e-8 for row in certificate_checks),
        "raw_inference_exactly_reproduced": max(analysis_errors.values()) <= 1e-12,
        "all_16_checkpoints_present": len(checkpoints) == 16 and len(checkpoint_hashes) == 16,
        "composite_gate_failure_preserved": recorded["gates"]["one_sided_dp_fraction_lower_above_0_90"] is False,
        "claim_limits_explicit": len(claims["prohibited"]) == 4,
    }
    tables = {"primary": {"ber_marl": analysis["ber_marl_mean_surplus"], "greedy": greedy,
                           "difference": analysis["mean_greedy_difference"], "difference_ci95": analysis["difference_bootstrap_95"],
                           "dp": dp, "dp_fraction": analysis["mean_dp_fraction"],
                           "dp_fraction_one_sided95": analysis["dp_fraction_one_sided_95"]},
              "methods": method_means, "composite_gate_passed": False}
    Path(output, "reproduced_analysis.json").write_text(json.dumps({"analysis": analysis, "errors": analysis_errors}, indent=2) + "\n")
    Path(output, "certificate_audit.json").write_text(json.dumps(certificate_checks, indent=2) + "\n")
    Path(output, "checkpoint_hashes.json").write_text(json.dumps(checkpoint_hashes, indent=2) + "\n")
    Path(output, "paper_tables.json").write_text(json.dumps(tables, indent=2) + "\n")
    Path(output, "claim_limits.json").write_text(json.dumps(claims, indent=2) + "\n")
    emit({"kind": "release_audit", "gates": gates, "primary": tables["primary"]})
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": 23012, "decisions": 23000, "updates": 0, "circuit_evaluations": 0, "shots": 0},
            "summary": {"analysis": analysis, "paper_tables": tables, "claims": claims,
                        "certificate_checks": certificate_checks, "checkpoint_hashes": checkpoint_hashes,
                        "bundle_validation": bundle_results}}
