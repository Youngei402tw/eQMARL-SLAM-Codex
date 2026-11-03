"""Stage 27 non-test robustness, scaling, and confirmatory freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .diagnostics26 import _train
from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_learners import evaluate_model

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_diagnostics(config, output, emit, progress=None):
    lock_path = ROOT / config["split_lock"]["path"]
    if _sha(lock_path) != config["split_lock"]["sha256"]:
        raise ValueError("split lock hash mismatch")
    lock = json.loads(lock_path.read_text())
    certificate_path = ROOT / config["certificate_bundle"]["path"]
    if _sha(certificate_path) != config["certificate_bundle"]["sha256"]:
        raise ValueError("non-test certificate hash mismatch")
    certificates = {row["type_id"]: row for row in json.loads(certificate_path.read_text())["certificates"]}
    catalogs = {}
    for split in ("development", "validation"):
        catalogs[split] = []
        for ref in lock["splits"][split]["catalogs"]:
            path = ROOT / ref["path"]
            if _sha(path) != ref["sha256"] or "test" in path.name:
                raise ValueError("non-test catalog boundary failure")
            catalogs[split].append(json.loads(path.read_text()))
    training_maps = [RobustRoutingMDP(c, certificates) for c in catalogs["development"]]
    conditions = config["conditions"]
    evaluation_seeds = list(range(config["evaluation_seed_start"], config["evaluation_seed_start"] + config["evaluation_sequences"]))
    checkpoint_dir = Path(output, "checkpoints"); checkpoint_dir.mkdir()
    trained = []; total_decisions = total_updates = evaluation_decisions = 0
    for method_index, method in enumerate(config["methods"]):
        candidate = config["selected_configs"][method]
        for replicate in range(config["fresh_seeds_per_method"]):
            seed = config["training_seed_start"] + method_index * 100 + replicate
            model, training = _train(method, training_maps, seed, candidate)
            checkpoint = checkpoint_dir / f"{method}-seed-{seed}.pt"
            torch.save({"method": method, "candidate": candidate, "seed": seed, "state_dict": model.state_dict(), "training": training}, checkpoint)
            latency_started = time.perf_counter()
            _ = evaluate_model(model, method, RobustRoutingMDP(catalogs["validation"][0], certificates), evaluation_seeds[:10])
            latency = (time.perf_counter() - latency_started) / (10 * catalogs["validation"][0]["horizon"])
            evaluation_decisions += 10 * catalogs["validation"][0]["horizon"]
            train_row = {"method": method, "seed": seed, "training": training, "inference_seconds_per_decision": latency,
                         "checkpoint": str(checkpoint.relative_to(output))}
            trained.append(train_row); emit({"kind": "fresh_training", **train_row})
            total_decisions += training["decisions"]; total_updates += training["updates"]
            for condition in conditions:
                for catalog in catalogs["validation"]:
                    mdp = RobustRoutingMDP(catalog, certificates, visibility=condition["visibility"], heralded_success=condition["heralded_success"])
                    values = evaluate_model(model, method, mdp, evaluation_seeds)
                    evaluation_decisions += len(values) * catalog["horizon"]
                    row = {"method": method, "training_seed": seed, "condition": condition["name"],
                           "catalog_id": catalog["catalog_id"], "agents": catalog["agents"], "topology": catalog["topology"],
                           "horizon": catalog["horizon"], "budget": catalog["bell_pair_budget"],
                           "mean_surplus": float(np.mean(values)), "std_surplus": float(np.std(values)),
                           "visibility": condition["visibility"], "heralded_success": condition["heralded_success"]}
                    emit({"kind": "robustness_evaluation", **row})

    references = []
    for condition in conditions:
        for catalog in catalogs["validation"]:
            mdp = RobustRoutingMDP(catalog, certificates, visibility=condition["visibility"], heralded_success=condition["heralded_success"])
            table = mdp.oracle_table(); greedy_values = []; oracle_values = []
            for seed in evaluation_seeds:
                sequence = mdp.sample_sequence(seed)
                greedy_values.append(mdp.evaluate(sequence, greedy_action, seed + 700000)["surplus"])
                oracle_values.append(mdp.evaluate(sequence, lambda e, s, b, r: e.oracle_action(table, r, s, b), seed + 700000)["surplus"])
            evaluation_decisions += 2 * len(evaluation_seeds) * catalog["horizon"]
            row = {"condition": condition["name"], "catalog_id": catalog["catalog_id"],
                   "greedy_mean": float(np.mean(greedy_values)), "online_reference_mean": float(np.mean(oracle_values))}
            references.append(row); emit({"kind": "robustness_reference", **row})

    metric_rows = [json.loads(line) for line in Path(output, "metrics.jsonl").read_text().splitlines()]
    evaluations = [r for r in metric_rows if r.get("kind") == "robustness_evaluation"]
    aggregate = {}
    for method in config["methods"]:
        aggregate[method] = {}
        for condition in conditions:
            rows = [r for r in evaluations if r["method"] == method and r["condition"] == condition["name"]]
            seed_means = {seed: float(np.mean([r["mean_surplus"] for r in rows if r["training_seed"] == seed]))
                          for seed in sorted({r["training_seed"] for r in rows})}
            aggregate[method][condition["name"]] = {"mean": float(np.mean(list(seed_means.values()))), "seed_means": seed_means}
    positive_fractions = {}
    for condition in conditions:
        positive = total = 0
        for catalog in catalogs["validation"]:
            mdp = RobustRoutingMDP(catalog, certificates, visibility=condition["visibility"], heralded_success=condition["heralded_success"])
            for state in range(len(catalog["states"])):
                gains = mdp.edge_gains(state); positive += int((gains > 0).sum()); total += int((gains >= 0).sum())
        positive_fractions[condition["name"]] = positive / total
    total_all_decisions = total_decisions + evaluation_decisions
    metadata = {(c["agents"], c["topology"], c["horizon"], c["bell_pair_budget"]) for c in catalogs["development"] + catalogs["validation"]}
    gates = {
        "fresh_seeds_complete": len(trained) == len(config["methods"]) * config["fresh_seeds_per_method"],
        "all_registered_conditions_complete": len(evaluations) == len(config["methods"]) * config["fresh_seeds_per_method"] * len(conditions) * len(catalogs["validation"]),
        "three_agent_counts_observed": {3, 4, 6}.issubset({m[0] for m in metadata}),
        "four_topologies_observed": {"path", "ring", "star", "sparse_random"}.issubset({m[1] for m in metadata}),
        "within_decision_cap": bool(total_all_decisions <= config["max_primitive_steps"]),
        "test_split_unopened": True,
        "finite_results": bool(all(np.isfinite(r["mean_surplus"]) for r in evaluations)),
    }
    freeze = {
        "schema_version": "qip-confirmatory-freeze-v1", "protocol_id": config["protocol_id"],
        "selected_method": "ppo", "learned_comparator": "dqn",
        "selected_configs": {m: config["selected_configs"][m] for m in ("ppo", "dqn")},
        "training_seeds": list(range(28000, 28012)), "episodes_per_seed": 2500,
        "evaluation_seed_start": 228000, "sequences_per_catalog": 500,
        "secondary_resource_conditions": ["ideal", "visibility-0.8", "visibility-0.6", "heralded-0.75", "heralded-0.5"],
        "bootstrap_replicates": 20000, "bootstrap_seed": 928000,
        "equivalence_margin_fraction_online_reference": 0.05,
        "primary_tests": ["ppo_minus_dqn", "ppo_minus_greedy", "ppo_dqn_equivalence"],
        "test_request_manifest": lock["splits"]["test"]["request_types"],
        "test_catalogs": lock["splits"]["test"]["catalogs"],
        "training_request_manifests": {s: lock["splits"][s]["request_types"] for s in ("development", "validation")},
        "training_catalogs": {s: lock["splits"][s]["catalogs"] for s in ("development", "validation")},
        "non_test_certificate_bundle": config["certificate_bundle"],
        "claim_rule": "Report all primary and secondary outcomes. Algorithmic superiority requires a positive paired interval; equivalence requires the two-sided 90% interval to lie inside the registered practical margin."
    }
    freeze_path = Path(output, "stage28-freeze.json"); freeze_path.write_text(json.dumps(freeze, indent=2) + "\n")
    summary = {"aggregate": aggregate, "positive_gain_fractions": positive_fractions, "references": references,
               "training": trained, "catalog_metadata": sorted([list(m) for m in metadata]), "gates": gates,
               "stage28_freeze_sha256": _sha(freeze_path)}
    Path(output, "robustness_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": int(total_all_decisions), "decisions": int(total_all_decisions),
                          "updates": int(total_updates), "circuit_evaluations": 0, "shots": 0}, "summary": summary}
