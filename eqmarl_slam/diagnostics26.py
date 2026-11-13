"""Stage 26 equal-budget learned baselines and mechanism ablations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .budgeted_entanglement import certify_request
from .robust_routing import RobustRoutingMDP, greedy_action
from .routing_learners import evaluate_model, train_ppo, train_q

ROOT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _train(method, maps, seed, candidate, feature_mode="full"):
    common = dict(episodes=candidate["episodes"], hidden=candidate["hidden"], learning_rate=candidate["learning_rate"], feature_mode=feature_mode)
    if method == "dqn":
        return train_q(maps, seed, double=False, **common)
    if method == "double_dqn":
        return train_q(maps, seed, double=True, **common)
    if method == "ppo":
        return train_ppo(maps, seed, recurrent=False, rollout_episodes=candidate.get("rollout_episodes", 16), epochs=candidate.get("epochs", 4), **common)
    if method == "recurrent_ppo":
        return train_ppo(maps, seed, recurrent=True, rollout_episodes=candidate.get("rollout_episodes", 16), epochs=candidate.get("epochs", 4), **common)
    raise ValueError(method)


def _evaluate(model, method, maps, seeds, feature_mode="full"):
    by_catalog = {}
    for mdp in maps:
        by_catalog[mdp.catalog["catalog_id"]] = evaluate_model(model, method, mdp, seeds, feature_mode)
    score = float(np.mean([np.mean(values) for values in by_catalog.values()]))
    return score, by_catalog


def _recovery(config, output, emit):
    prior = ROOT / config["recovery_from"]
    manifest = json.loads((prior / "manifest.json").read_text())
    if manifest["status"] != "failed" or manifest["source_commit"] != config["failed_source_commit"]:
        raise ValueError("recovery source identity mismatch")
    metrics_path = prior / "metrics.jsonl"
    if _sha(metrics_path) != config["failed_metrics_sha256"]:
        raise ValueError("failed-run metrics hash mismatch")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    trials = [row for row in rows if row.get("kind") == "tuning_trial"]
    ablations = [row for row in rows if row.get("kind") == "ablation"]
    throughput = [row for row in rows if row.get("kind") == "throughput"]
    methods = config["methods"]; proposed = config["proposed_method"]
    method_scores, selections = {}, {}
    for method in methods:
        candidate_names = sorted({row["candidate"] for row in trials if row["method"] == method})
        candidate_scores = {name: float(np.mean([row["validation_score"] for row in trials
                                                  if row["method"] == method and row["candidate"] == name]))
                            for name in candidate_names}
        selected = max(candidate_scores, key=candidate_scores.get)
        selections[method] = selected; method_scores[method] = candidate_scores[selected]
    strongest = max((m for m in methods if m != proposed), key=lambda m: method_scores[m])
    ablation_scores = {mode: float(np.mean([row["validation_score"] for row in ablations if row["feature_mode"] == mode]))
                       for mode in sorted({row["feature_mode"] for row in ablations})}
    training_decisions = sum(int(row["training"]["decisions"]) for row in trials + ablations)
    training_updates = sum(int(row["training"]["updates"]) for row in trials + ablations)
    prior_wall = float(manifest["resources_consumed"]["wall_seconds"])
    reconstructed_prior = {"wall_seconds": prior_wall, "cpu_core_hours": manifest["resources_consumed"]["cpu_core_hours"],
                           "training_decisions": training_decisions, "training_updates": training_updates,
                           "certification_updates": 30 * 16 * 800, "metrics_rows": len(rows), "checkpoints": len(list((prior / "checkpoints").glob("*.pt")))}
    gates = {
        "failed_bundle_identity_verified": True,
        "all_tuning_trials_recovered": len(trials) == len(methods) * 3 * 2,
        "all_ablation_trials_recovered": len(ablations) == 4 * 2,
        "all_checkpoints_retained": reconstructed_prior["checkpoints"] == 32,
        "finite_scores": bool(all(np.isfinite(row["validation_score"]) for row in trials + ablations)),
        "test_split_unopened": True,
    }
    summary = {"development_outcome": "null_proposed_method_advantage", "method_scores": method_scores,
               "selected_candidates": selections, "proposed_method": proposed, "strongest_learned_baseline": strongest,
               "proposed_minus_baseline_validation": float(method_scores[proposed] - method_scores[strongest]),
               "ablation_scores": ablation_scores, "throughput": throughput, "reconstructed_prior_resources": reconstructed_prior,
               "failed_bundle": config["recovery_from"], "failed_metrics_sha256": config["failed_metrics_sha256"], "gates": gates}
    Path(output, "recovered_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit({"kind": "recovery_summary", **summary})
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": 0, "decisions": 0, "updates": 0, "circuit_evaluations": 0, "shots": 0},
            "summary": summary}


def run_diagnostics(config, output, emit, progress=None):
    if config.get("recovery_from"):
        return _recovery(config, output, emit)
    lock_path = ROOT / config["split_lock"]["path"]
    if _sha(lock_path) != config["split_lock"]["sha256"]:
        raise ValueError("split lock hash mismatch")
    lock = json.loads(lock_path.read_text())
    allowed = ("development", "validation")
    if any(lock["splits"][split]["request_types"]["path"].find("test") >= 0 for split in allowed):
        raise ValueError("test path entered non-test stage")

    certificates = {}
    certification_updates = 0
    for split in allowed:
        reference = lock["splits"][split]["request_types"]
        path = ROOT / reference["path"]
        if _sha(path) != reference["sha256"]:
            raise ValueError(f"{split} request hash mismatch")
        for index, request in enumerate(json.loads(path.read_text())["types"]):
            row = certify_request(request, config["certificate_seed"] + len(certificates) * 32)
            row["split"] = split; certificates[row["type_id"]] = row
            certification_updates += 16 * 800
            emit({"kind": "request_certificate", **row})
    Path(output, "non_test_certificates.json").write_text(json.dumps({"certificates": list(certificates.values())}, indent=2) + "\n")

    maps = {}
    for split in allowed:
        maps[split] = []
        for reference in lock["splits"][split]["catalogs"]:
            path = ROOT / reference["path"]
            if _sha(path) != reference["sha256"]:
                raise ValueError(f"{split} catalog hash mismatch")
            maps[split].append(RobustRoutingMDP(json.loads(path.read_text()), certificates))

    throughput = []
    for index, method in enumerate(config["methods"]):
        candidate = dict(config["candidates"][method][0]); candidate["episodes"] = config["throughput_episodes"]
        started = time.perf_counter(); _, log = _train(method, [maps["development"][2]], config["tuning_seed_start"] + 900 + index, candidate)
        elapsed = time.perf_counter() - started
        row = {"method": method, "episodes": candidate["episodes"], "decisions": log["decisions"], "wall_seconds": elapsed,
               "decisions_per_second": log["decisions"] / elapsed}
        throughput.append(row); emit({"kind": "throughput", **row})
    projected = sum((config["candidates"][m][0]["episodes"] * 12 / max(1e-9, throughput[i]["decisions_per_second"]))
                    * len(config["candidates"][m]) * config["seeds_per_candidate"] for i, m in enumerate(config["methods"]))
    if projected > config["wall_seconds"] * 0.8:
        raise RuntimeError("measured throughput cannot fit registered wall budget")

    eval_seeds = list(range(config["validation_seed_start"], config["validation_seed_start"] + config["validation_sequences"]))
    checkpoint_dir = Path(output, "checkpoints"); checkpoint_dir.mkdir()
    trials = []; total_decisions = total_updates = 0
    for method_index, method in enumerate(config["methods"]):
        for candidate_index, candidate in enumerate(config["candidates"][method]):
            for replicate in range(config["seeds_per_candidate"]):
                seed = config["tuning_seed_start"] + method_index * 100 + candidate_index * 10 + replicate
                model, train = _train(method, maps["development"], seed, candidate)
                score, by_catalog = _evaluate(model, method, maps["validation"], eval_seeds)
                checkpoint = checkpoint_dir / f"{method}-{candidate['name']}-seed-{seed}.pt"
                torch.save({"method": method, "candidate": candidate, "seed": seed, "state_dict": model.state_dict(), "train": train}, checkpoint)
                row = {"method": method, "candidate": candidate["name"], "seed": seed, "validation_score": score,
                       "catalog_means": {key: float(np.mean(value)) for key, value in by_catalog.items()}, "training": train,
                       "checkpoint": str(checkpoint.relative_to(output))}
                trials.append(row); emit({"kind": "tuning_trial", **row})
                total_decisions += train["decisions"] + len(maps["validation"]) * len(eval_seeds) * np.mean([m.catalog["horizon"] for m in maps["validation"]])
                total_updates += train["updates"]

    selections = {}; method_scores = {}
    for method in config["methods"]:
        candidate_scores = {}
        for candidate in config["candidates"][method]:
            rows = [r for r in trials if r["method"] == method and r["candidate"] == candidate["name"]]
            candidate_scores[candidate["name"]] = float(np.mean([r["validation_score"] for r in rows]))
        selected_name = max(candidate_scores, key=candidate_scores.get)
        selections[method] = next(c for c in config["candidates"][method] if c["name"] == selected_name)
        method_scores[method] = candidate_scores[selected_name]

    ablations = []
    proposed = config["proposed_method"]
    for mode_index, mode in enumerate(config["ablations"]):
        for replicate in range(config["ablation_seeds"]):
            seed = config["ablation_seed_start"] + mode_index * 10 + replicate
            model, train = _train(proposed, maps["development"], seed, selections[proposed], mode)
            score, by_catalog = _evaluate(model, proposed, maps["validation"], eval_seeds, mode)
            checkpoint = checkpoint_dir / f"ablation-{mode}-seed-{seed}.pt"
            torch.save({"method": proposed, "feature_mode": mode, "seed": seed, "state_dict": model.state_dict(), "train": train}, checkpoint)
            row = {"feature_mode": mode, "seed": seed, "validation_score": score,
                   "catalog_means": {key: float(np.mean(value)) for key, value in by_catalog.items()}, "training": train,
                   "checkpoint": str(checkpoint.relative_to(output))}
            ablations.append(row); emit({"kind": "ablation", **row})
            total_decisions += train["decisions"] + len(maps["validation"]) * len(eval_seeds) * np.mean([m.catalog["horizon"] for m in maps["validation"]])
            total_updates += train["updates"]

    baseline = {}
    for mdp in maps["validation"]:
        table = mdp.oracle_table(); greedy_values = []; oracle_values = []
        for seed in eval_seeds:
            sequence = mdp.sample_sequence(seed)
            greedy_values.append(mdp.evaluate(sequence, greedy_action, seed=seed)["surplus"])
            oracle_values.append(mdp.evaluate(sequence, lambda e, s, b, r: e.oracle_action(table, r, s, b), seed=seed)["surplus"])
        baseline[mdp.catalog["catalog_id"]] = {"greedy": float(np.mean(greedy_values)), "online_reference": float(np.mean(oracle_values))}

    strongest_baseline = max((m for m in config["methods"] if m != proposed), key=lambda m: method_scores[m])
    gates = {
        "test_split_unopened": True,
        "four_methods_completed": set(method_scores) == set(config["methods"]),
        "equal_tuning_trials": all(sum(r["method"] == m for r in trials) == len(config["candidates"][m]) * config["seeds_per_candidate"] for m in config["methods"]),
        "all_feature_ablations_completed": len(ablations) == len(config["ablations"]) * config["ablation_seeds"],
        "finite_validation_scores": all(np.isfinite(r["validation_score"]) for r in trials + ablations),
        "within_decision_cap": bool(total_decisions <= config["max_primitive_steps"]),
    }
    summary = {"throughput": throughput, "projected_tuning_wall_seconds": projected, "method_scores": method_scores,
               "selected_configs": selections, "proposed_method": proposed, "strongest_learned_baseline": strongest_baseline,
               "proposed_minus_baseline_validation": method_scores[proposed] - method_scores[strongest_baseline],
               "ablations": ablations, "baselines": baseline, "gates": gates}
    Path(output, "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"passed": all(gates.values()), "gates": gates,
            "resources": {"primitive_steps": int(total_decisions), "decisions": int(total_decisions),
                          "updates": int(total_updates + certification_updates), "circuit_evaluations": certification_updates * 9, "shots": 0},
            "summary": summary}
