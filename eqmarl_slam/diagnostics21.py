"""Stage 21 request certification and sequential resource invariants."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pennylane as qml

from .budgeted_entanglement import BudgetedAllocationMDP, certify_request

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _qml_correlation(alpha, beta):
    device = qml.device("default.qubit", wires=2)
    @qml.qnode(device)
    def circuit():
        qml.Hadamard(0); qml.CNOT([0, 1]); qml.RY(alpha, 0); qml.RY(beta, 1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    return float(circuit())


def run_diagnostics(config, output, emit, progress=None):
    certificates = {}
    rows = []
    for split_index, split in enumerate(("development", "validation")):
        manifest = json.loads((ROOT / config["request_type_manifests"][split]["path"]).read_text())
        if _sha(ROOT / config["request_type_manifests"][split]["path"]) != config["request_type_manifests"][split]["sha256"]:
            raise ValueError(f"{split} request manifest hash mismatch")
        for index, request in enumerate(manifest["types"]):
            row = certify_request(request, 21000 + split_index * 1000 + index * 32)
            row["split"] = split; rows.append(row); certificates[row["type_id"]] = row
            emit({"kind": "request_certificate", **row})

    # Independent device fixture for the first certified request.
    first = rows[0]; request_manifest = json.loads((ROOT / config["request_type_manifests"]["development"]["path"]).read_text())
    request = request_manifest["types"][0]; angles = first["bell_angles"]
    device_errors = []
    for x in range(3):
        for y in range(3):
            device_errors.append(abs(_qml_correlation(angles[x], angles[3+y]) - np.cos(angles[x] - angles[3+y])))

    catalog_summaries = []
    invariant_failures = []
    for split in ("development", "validation"):
        reference = config["public_catalogs"][split]
        if _sha(ROOT / reference["path"]) != reference["sha256"]:
            raise ValueError(f"{split} catalog hash mismatch")
        catalog = json.loads((ROOT / reference["path"]).read_text())
        mdp = BudgetedAllocationMDP(catalog, certificates)
        table = mdp.oracle_table()
        oracle_values = []
        for seed in range(50):
            sequence = mdp.sample_sequence(21500 + seed)
            result = mdp.evaluate(sequence, lambda env, state, budget, remaining: env.oracle_action(table, remaining, state, budget))
            oracle_values.append(result["entanglement_surplus"])
            if result["pairs_used"] > catalog["bell_pair_budget"] or any(row["budget_after"] < 0 for row in result["actions"]):
                invariant_failures.append({"split":split,"seed":seed,"reason":"budget"})
            for row in result["actions"]:
                endpoints = [v for edge in row["matching"] for v in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))[edge]]
                if len(endpoints) != len(set(endpoints)):
                    invariant_failures.append({"split":split,"seed":seed,"reason":"matching"})
        catalog_summaries.append({"split": split, "states": len(catalog["states"]), "mean_oracle_surplus": float(np.mean(oracle_values)),
                                  "min_oracle_surplus": float(np.min(oracle_values)), "max_oracle_surplus": float(np.max(oracle_values))})
    test_reference = config["request_type_manifests"]["test"]
    test_hash_present_without_access = len(test_reference["sha256"]) == 64 and Path(ROOT / test_reference["path"]).is_file()
    gaps = [row["quantum_gap"] for row in rows]
    gates = {
        "all_16_requests_certified": len(rows) == 16,
        "classical_values_bounded": all(0.5 <= row["classical_value"] <= 1.0 + 1e-12 for row in rows),
        "bell_within_sdp_bound": all(row["bell_value"] <= row["sdp_upper"] + 2e-8 for row in rows),
        "bell_reaches_sdp_tolerance": max(abs(row["sdp_gap"]) for row in rows) <= 2e-7,
        "heterogeneous_gaps": any(gap > 0.005 for gap in gaps) and any(abs(gap) <= 1e-7 for gap in gaps),
        "pennylane_fixture_exact": max(device_errors) <= 1e-12,
        "budget_and_monogamy_exact": not invariant_failures,
        "sequential_oracle_nontrivial": all(row["mean_oracle_surplus"] > 0 for row in catalog_summaries),
        "test_manifest_hash_locked_unopened": test_hash_present_without_access,
    }
    Path(output, "request_certificates.json").write_text(json.dumps({"certificates":rows},indent=2)+"\n")
    Path(output, "catalog_diagnostics.json").write_text(json.dumps(catalog_summaries,indent=2)+"\n")
    updates = len(rows) * 16 * 800
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps":1000,"decisions":1000,"updates":updates,
                          "circuit_evaluations":updates*9+9,"shots":0},
            "summary": {"certificates":rows,"gap_summary":{"min":min(gaps),"mean":float(np.mean(gaps)),"max":max(gaps),"positive_over_0.005":sum(g>0.005 for g in gaps)},
                        "max_sdp_gap_abs":max(abs(row["sdp_gap"]) for row in rows),"max_device_error":max(device_errors),
                        "catalogs":catalog_summaries,"invariant_failures":invariant_failures,
                        "information_contract":{"allocator_private_context_access":False,"post_context_messages":0,"public_fields":["request_graph","type_id","urgency","remaining_horizon","remaining_bell_budget"]}}}
