"""Stage 29 independent release audit for the QIP robust entanglement-routing study.

This module re-derives every reportable Stage 28 quantity from raw episode rows,
committed manifests, and checkpoints. It deliberately does not import the Stage 28
analysis code (``diagnostics28``), the routing environment, or the learners: it
uses its own classical enumeration, its own XOR-game SDP, its own Bell evaluation
at the recorded angles, its own bootstrap and Holm implementations, and its own
resource arithmetic. It performs no allocator decisions and no training.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .archive import validate_bundle

ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-9


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path: Path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------------------
# Independent statistics
# ---------------------------------------------------------------------------

def percentile_bootstrap(values, seed: int, replicates: int, tail: float) -> list[float]:
    """Seed-level percentile bootstrap of the mean using the registered resampling stream."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.fromiter((values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(replicates)),
                        dtype=float, count=replicates)
    return [float(np.percentile(means, tail)), float(np.percentile(means, 100.0 - tail))]


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at one)."""
    m = len(p_values)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def exact_signed_rank_greater(differences) -> float:
    """Exact one-sided Wilcoxon signed-rank p-value for H1: median difference > 0 (no zeros/ties assumed)."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    if len(set(np.abs(d))) != n:
        raise ValueError("tied magnitudes require the normal approximation")
    observed = int(ranks[d > 0].sum())
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        counts[rank:] = counts[rank:] + counts[:-rank].copy()
    return float(counts[observed:].sum() / 2 ** n)


# ---------------------------------------------------------------------------
# Independent request certification
# ---------------------------------------------------------------------------

def classical_optimum(request: dict) -> float:
    predicate = np.asarray(request["predicate"], dtype=int)
    weights = np.asarray(request["weights"], dtype=float)
    weights = weights / weights.sum()
    best = 0.0
    for alice in itertools.product((0, 1), repeat=3):
        for bob in itertools.product((0, 1), repeat=3):
            win = sum(weights[x, y] for x in range(3) for y in range(3) if (alice[x] ^ bob[y]) == predicate[x, y])
            best = max(best, win)
    return float(best)


def bell_value_at_angles(request: dict, angles: list[float]) -> float:
    """Win probability of |Phi+> with real-plane projective measurements at the recorded angles."""
    predicate = np.asarray(request["predicate"], dtype=int)
    weights = np.asarray(request["weights"], dtype=float)
    weights = weights / weights.sum()
    value = 0.0
    for x in range(3):
        for y in range(3):
            correlation = math.cos(angles[x] - angles[3 + y])
            sign = 1.0 - 2.0 * predicate[x, y]
            value += weights[x, y] * 0.5 * (1.0 + sign * correlation)
    return float(value)


def xor_quantum_upper(request: dict) -> float:
    """Tsirelson-type SDP value for the XOR game, written independently of Stage 21 code."""
    import cvxpy as cp

    predicate = np.asarray(request["predicate"], dtype=int)
    weights = np.asarray(request["weights"], dtype=float)
    weights = weights / weights.sum()
    signs = 1.0 - 2.0 * predicate
    vectors = cp.Variable((6, 6), PSD=True)
    bias = sum(weights[x, y] * signs[x, y] * vectors[x, 3 + y] for x in range(3) for y in range(3))
    problem = cp.Problem(cp.Maximize(bias), [cp.diag(vectors) == 1])
    problem.solve(solver="CLARABEL")
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(problem.status)
    return float((1.0 + problem.value) / 2.0)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_diagnostics(config, output, emit, progress=None):
    output = Path(output)
    bundle = ROOT / config["confirmatory_bundle"]
    freeze_path = ROOT / config["freeze"]["path"]
    checks: dict[str, bool] = {}
    findings: list[dict] = []

    # 1. Archive and identity -------------------------------------------------
    bundle_checks = [validate_bundle(ROOT / path, repository_root=ROOT) for path in config["bundles"]]
    checks["all_bundles_valid"] = all(item.get("valid") for item in bundle_checks)
    checks["freeze_hash_matches"] = _sha(freeze_path) == config["freeze"]["sha256"]
    freeze = json.loads(freeze_path.read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    recorded = json.loads((bundle / "confirmatory_analysis.json").read_text())
    resolved = (bundle / "resolved_config.yaml").read_text()
    checks["bundle_source_is_frozen_commit"] = manifest["source_commit"] == config["stage28_source_commit"]
    checks["bundle_status_completed"] = manifest["status"] == "completed"
    checks["bundle_config_hash_matches_committed_config"] = (
        _sha(ROOT / config["stage28_config"]) == manifest["config_sha256"] == hashlib.sha256(resolved.encode()).hexdigest())
    emit({"kind": "identity_audit", "checks": {k: checks[k] for k in list(checks)}})

    # 2. Raw rows -------------------------------------------------------------
    baseline = defaultdict(list)                     # (method, catalog) -> [(sequence, surplus)]
    learned = defaultdict(dict)                      # (method, condition, catalog, seed) -> {sequence: surplus}
    pair_rows = []
    nonfinite = 0
    for row in _rows(bundle / "episode_results.jsonl"):
        if not math.isfinite(row["surplus"]):
            nonfinite += 1
        if row["kind"] == "baseline_episode":
            baseline[(row["method"], row["catalog_id"])].append((row["sequence_seed"], row["surplus"]))
        elif row["kind"] == "learned_episode":
            key = (row["method"], row["condition"], row["catalog_id"], row["training_seed"])
            if row["sequence_seed"] in learned[key]:
                raise ValueError(f"duplicate learned row {key} {row['sequence_seed']}")
            learned[key][row["sequence_seed"]] = row["surplus"]
            pair_rows.append((row["catalog_id"], row["pairs_attempted"], row["pairs_unused"], row["surplus"]))
        else:
            raise ValueError(f"unexpected raw row kind {row['kind']}")
    catalogs = [ref["catalog_id"] for ref in freeze["test_catalogs"]]
    catalog_data = {ref["catalog_id"]: json.loads((ROOT / ref["path"]).read_text()) for ref in freeze["test_catalogs"]}
    seeds = list(freeze["training_seeds"])
    primary = list(range(freeze["evaluation_seed_start"], freeze["evaluation_seed_start"] + freeze["sequences_per_catalog"]))
    secondary = primary[: config["secondary_sequences_per_catalog"]]
    noisy_conditions = [c for c in freeze["secondary_resource_conditions"] if c != "ideal"]
    baseline_methods = ["classical_exact", "random", "static", "greedy", "online_reference", "future_clairvoyant", "message_oracle"]

    completeness = {
        "baseline_cells": all(sorted(s for s, _ in baseline[(m, c)]) == primary for m in baseline_methods for c in catalogs),
        "primary_cells": all(sorted(learned[(m, "ideal", c, s)]) == primary for m in ("dqn", "ppo") for c in catalogs for s in seeds),
        "secondary_cells": all(sorted(learned[("ppo", cond, c, s)]) == secondary for cond in noisy_conditions for c in catalogs for s in seeds),
        "no_secondary_dqn_rows": not any(k[0] == "dqn" and k[1] != "ideal" for k in learned),
        "row_count": sum(len(v) for v in baseline.values()) + sum(len(v) for v in learned.values()),
        "expected_row_count": (len(baseline_methods) * len(catalogs) * len(primary) + 2 * len(seeds) * len(catalogs) * len(primary)
                               + len(seeds) * len(noisy_conditions) * len(catalogs) * len(secondary)),
        "nonfinite_surplus_rows": nonfinite,
    }
    checks["raw_rows_complete_and_paired"] = (completeness["baseline_cells"] and completeness["primary_cells"]
                                              and completeness["secondary_cells"] and completeness["no_secondary_dqn_rows"]
                                              and completeness["row_count"] == completeness["expected_row_count"] and nonfinite == 0)
    emit({"kind": "raw_row_audit", **completeness})

    # 3. Reproduce the registered analysis ------------------------------------
    def method_catalog_mean(method, catalog):
        return float(np.mean([v for _, v in baseline[(method, catalog)]]))

    baseline_table = {m: {c: method_catalog_mean(m, c) for c in catalogs} for m in baseline_methods}
    seed_catalog = {m: {s: {c: float(np.mean(list(learned[(m, "ideal", c, s)].values()))) for c in catalogs} for s in seeds}
                    for m in ("dqn", "ppo")}
    seed_mean = {m: np.asarray([np.mean([seed_catalog[m][s][c] for c in catalogs]) for s in seeds]) for m in ("dqn", "ppo")}
    method_means = {"ppo": float(seed_mean["ppo"].mean()), "dqn": float(seed_mean["dqn"].mean()),
                    **{m: float(np.mean(list(baseline_table[m].values()))) for m in baseline_methods}}
    greedy_mean, online_mean = method_means["greedy"], method_means["online_reference"]
    difference = seed_mean["ppo"] - seed_mean["dqn"]
    versus_greedy = seed_mean["ppo"] - greedy_mean
    reps, boot = freeze["bootstrap_replicates"], freeze["bootstrap_seed"]
    diff95 = percentile_bootstrap(difference, boot, reps, 2.5)
    diff90 = percentile_bootstrap(difference, boot + 1, reps, 5.0)
    greedy95 = percentile_bootstrap(versus_greedy, boot + 2, reps, 2.5)
    margin = freeze["equivalence_margin_fraction_online_reference"] * online_mean
    tests = {"ppo_superior_to_dqn": bool(diff95[0] > 0), "ppo_superior_to_greedy": bool(greedy95[0] > 0),
             "ppo_dqn_equivalent_within_5pct_online_reference": bool(-margin < diff90[0] and diff90[1] < margin)}
    catalog_tests = []
    for c in catalogs:
        a = np.asarray([seed_catalog["ppo"][s][c] for s in seeds]); b = np.asarray([seed_catalog["dqn"][s][c] for s in seeds])
        catalog_tests.append({"catalog_id": c, "ppo_mean": float(a.mean()), "dqn_mean": float(b.mean()),
                              "mean_difference": float((a - b).mean()), "ppo_wins": int((a > b).sum()),
                              "raw_p": exact_signed_rank_greater(a - b)})
    for row, adjusted in zip(catalog_tests, holm([r["raw_p"] for r in catalog_tests])):
        row["holm_p"] = adjusted
    secondary_seed = {cond: {s: float(np.mean([np.mean(list(learned[("ppo", cond, c, s)].values())) for c in catalogs]))
                             for s in seeds} for cond in noisy_conditions}
    secondary_means = {cond: float(np.mean(list(v.values()))) for cond, v in secondary_seed.items()}
    # Ideal PPO on the same first 100 sequences, for a matched comparison with the noisy conditions.
    ideal_first100 = float(np.mean([np.mean([np.mean([learned[("ppo", "ideal", c, s)][q] for q in secondary]) for c in catalogs]) for s in seeds]))

    comparisons = []

    def compare(name, mine, theirs):
        mine_flat, theirs_flat = np.ravel(np.asarray(mine, dtype=float)), np.ravel(np.asarray(theirs, dtype=float))
        ok = mine_flat.shape == theirs_flat.shape and bool(np.all(np.abs(mine_flat - theirs_flat) <= TOLERANCE))
        comparisons.append({"quantity": name, "reproduced": np.asarray(mine).tolist(), "recorded": np.asarray(theirs).tolist(), "match": ok})

    for m in ("ppo", "dqn", *baseline_methods):
        compare(f"method_mean:{m}", method_means[m], recorded["method_means"][m])
    compare("ppo_minus_dqn", difference.mean(), recorded["ppo_minus_dqn"])
    compare("ppo_minus_dqn_95", diff95, recorded["ppo_minus_dqn_95"])
    compare("ppo_minus_dqn_90", diff90, recorded["ppo_minus_dqn_90"])
    compare("ppo_minus_greedy", versus_greedy.mean(), recorded["ppo_minus_greedy"])
    compare("ppo_minus_greedy_95", greedy95, recorded["ppo_minus_greedy_95"])
    compare("equivalence_margin", margin, recorded["equivalence_margin"])
    for mine, theirs in zip(catalog_tests, recorded["catalog_tests"]):
        compare(f"catalog:{mine['catalog_id']}", [mine["ppo_mean"], mine["dqn_mean"], mine["raw_p"], mine["holm_p"]],
                [theirs["ppo_mean"], theirs["dqn_mean"], theirs["raw_p"], theirs["holm_p"]])
    for cond in noisy_conditions:
        compare(f"secondary:{cond}", secondary_means[cond], recorded["secondary_condition_means"][cond])
    tests_match = tests == recorded["scientific_tests"]
    checks["registered_analysis_reproduced"] = all(c["match"] for c in comparisons) and tests_match
    # Sensitivity: a fresh, differently seeded bootstrap should not change any decision.
    sensitivity = {"ppo_minus_dqn_95": percentile_bootstrap(difference, 29_000, reps, 2.5),
                   "ppo_minus_dqn_90": percentile_bootstrap(difference, 29_001, reps, 5.0),
                   "ppo_minus_greedy_95": percentile_bootstrap(versus_greedy, 29_002, reps, 2.5)}
    sensitivity_decisions = {"ppo_superior_to_dqn": sensitivity["ppo_minus_dqn_95"][0] > 0,
                             "ppo_superior_to_greedy": sensitivity["ppo_minus_greedy_95"][0] > 0,
                             "ppo_dqn_equivalent_within_5pct_online_reference": -margin < sensitivity["ppo_minus_dqn_90"][0] and sensitivity["ppo_minus_dqn_90"][1] < margin}
    checks["decisions_stable_under_alternate_bootstrap_stream"] = sensitivity_decisions == tests
    # The greedy contrast uses a deterministic baseline; also report the per-catalog paired PPO-greedy differences.
    per_catalog_vs_greedy = {c: {"ppo_minus_greedy": float(np.mean([seed_catalog["ppo"][s][c] for s in seeds]) - baseline_table["greedy"][c]),
                                 "dqn_minus_greedy": float(np.mean([seed_catalog["dqn"][s][c] for s in seeds]) - baseline_table["greedy"][c]),
                                 "ppo_seeds_above_greedy": int(sum(seed_catalog["ppo"][s][c] > baseline_table["greedy"][c] for s in seeds)),
                                 "dqn_seeds_above_greedy": int(sum(seed_catalog["dqn"][s][c] > baseline_table["greedy"][c] for s in seeds))}
                             for c in catalogs}
    fraction_online = {m: {c: (float(np.mean([seed_catalog[m][s][c] for s in seeds])) / baseline_table["online_reference"][c]) for c in catalogs}
                       for m in ("ppo", "dqn")}
    for m in ("random", "static", "greedy", "future_clairvoyant"):
        fraction_online[m] = {c: baseline_table[m][c] / baseline_table["online_reference"][c] for c in catalogs}
    reproduced = {"method_means": method_means, "baseline_by_catalog": baseline_table,
                  "learned_seed_catalog_means": {m: {str(s): v for s, v in d.items()} for m, d in seed_catalog.items()},
                  "learned_seed_means": {m: v.tolist() for m, v in seed_mean.items()},
                  "ppo_minus_dqn": float(difference.mean()), "ppo_minus_dqn_95": diff95, "ppo_minus_dqn_90": diff90,
                  "ppo_minus_greedy": float(versus_greedy.mean()), "ppo_minus_greedy_95": greedy95,
                  "ppo_seeds_above_dqn": int((difference > 0).sum()), "ppo_seeds_above_greedy": int((versus_greedy > 0).sum()),
                  "dqn_seeds_above_greedy": int((seed_mean["dqn"] > greedy_mean).sum()),
                  "online_reference_mean": online_mean, "equivalence_margin": margin, "scientific_tests": tests,
                  "catalog_tests": catalog_tests, "secondary_condition_means": secondary_means,
                  "secondary_seed_means": {k: {str(s): v for s, v in d.items()} for k, d in secondary_seed.items()},
                  "ppo_ideal_first_100_sequences": ideal_first100,
                  "per_catalog_versus_greedy": per_catalog_vs_greedy, "fraction_of_online_reference": fraction_online,
                  "learned_fraction_of_online_reference_overall": {m: method_means[m] / online_mean for m in ("ppo", "dqn")},
                  "bootstrap_sensitivity_alternate_stream": sensitivity, "comparisons": comparisons}
    (output / "reproduced_analysis.json").write_text(json.dumps(reproduced, indent=2) + "\n")
    emit({"kind": "analysis_reproduction", "matched": sum(c["match"] for c in comparisons), "total": len(comparisons),
          "tests_match": tests_match, "tests": tests})

    # 4. Certificates and visibility thresholds -------------------------------
    test_manifest = json.loads((ROOT / freeze["test_request_manifest"]["path"]).read_text())
    recorded_certs = {row["type_id"]: row for row in json.loads((bundle / "test_certificates.json").read_text())["certificates"]}
    certificate_rows = []
    for request in test_manifest["types"]:
        cert = recorded_certs[request["type_id"]]
        classical = classical_optimum(request)
        at_angles = bell_value_at_angles(request, cert["bell_angles"])
        upper = xor_quantum_upper(request)
        gap = at_angles - classical
        threshold = (classical - 0.5) / (at_angles - 0.5) if at_angles > 0.5 else 1.0
        certificate_rows.append({"type_id": request["type_id"], "classical_exact": classical, "classical_recorded": cert["classical_value"],
                                 "bell_at_recorded_angles": at_angles, "bell_recorded": cert["bell_value"], "sdp_upper_independent": upper,
                                 "sdp_upper_recorded": cert["sdp_upper"], "quantum_gain": gap,
                                 "positive_gain": bool(gap > 1e-7), "visibility_threshold": min(1.0, threshold)})
    cert_ok = all(abs(r["classical_exact"] - r["classical_recorded"]) < 1e-12 and abs(r["bell_at_recorded_angles"] - r["bell_recorded"]) < 1e-9
                  and r["bell_recorded"] <= r["sdp_upper_independent"] + 1e-6 and abs(r["sdp_upper_independent"] - r["sdp_upper_recorded"]) < 1e-6
                  and r["bell_recorded"] >= r["classical_recorded"] - 1e-9 for r in certificate_rows)
    thresholds = np.asarray([r["visibility_threshold"] for r in certificate_rows])
    threshold_summary = {"min": float(thresholds.min()), "median": float(np.median(thresholds)), "mean": float(thresholds.mean()), "max": float(thresholds.max())}
    compare("visibility_thresholds", [threshold_summary[k] for k in ("min", "median", "mean", "max")],
            [recorded["test_visibility_thresholds"][k] for k in ("min", "median", "mean", "max")])
    # Fraction of sealed test opportunities (request occurrences in test catalogs) with positive gain by visibility.
    occurrences = [(c, s, r["type_id"], r["urgency"]) for c in catalogs for s, state in enumerate(catalog_data[c]["states"]) for r in state["requests"]]
    by_type = {r["type_id"]: r for r in certificate_rows}
    visibility_grid = [round(v, 3) for v in np.linspace(0.80, 1.0, 41)]
    positive_fraction = {str(v): float(np.mean([0.5 + v * (by_type[t]["bell_at_recorded_angles"] - 0.5) - by_type[t]["classical_exact"] > 1e-12
                                                for _, _, t, _ in occurrences])) for v in visibility_grid}
    checks["certificates_independently_verified"] = cert_ok
    checks["visibility_thresholds_reproduced"] = comparisons[-1]["match"]
    certificate_summary = {"rows": certificate_rows, "threshold_summary": threshold_summary,
                           "positive_gain_types": int(sum(r["positive_gain"] for r in certificate_rows)), "types": len(certificate_rows),
                           "test_opportunities": len(occurrences), "positive_opportunity_fraction_by_visibility": positive_fraction,
                           "sdp_solver": "CLARABEL via cvxpy, independent formulation", "analytic_bell_evaluations": len(certificate_rows),
                           "sdp_solves": len(certificate_rows)}
    (output / "certificate_audit.json").write_text(json.dumps(certificate_summary, indent=2) + "\n")
    emit({"kind": "certificate_audit", "ok": cert_ok, **threshold_summary, "positive_gain_types": certificate_summary["positive_gain_types"]})

    # 5. Checkpoints ------------------------------------------------------------
    import torch

    inventory = {row["path"]: row for row in json.loads((bundle / "artifact_manifest.json").read_text())["artifacts"]}
    metrics = list(_rows(bundle / "metrics.jsonl"))
    seed_rows = {(r["method"], r["training_seed"]): r for r in metrics if r["kind"] == "confirmatory_seed"}
    checkpoint_rows = []
    for method in ("dqn", "ppo"):
        for seed in seeds:
            relative = f"checkpoints/{method}-seed-{seed}.pt"
            path = bundle / relative
            digest = _sha(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            parameters = int(sum(t.numel() for t in payload["state_dict"].values()))
            expected = dict(freeze["selected_configs"][method]); expected["episodes"] = freeze["episodes_per_seed"]
            training = payload["training"]
            checkpoint_rows.append({"path": relative, "sha256": digest, "inventory_sha256": inventory[relative]["sha256"],
                                    "method": payload["method"], "seed": payload["seed"], "parameters": parameters,
                                    "recorded_parameters": training["parameters"], "candidate_matches_freeze": payload["candidate"] == expected,
                                    "episodes": training["episodes"], "decisions": training["decisions"], "updates": training["updates"],
                                    "final_100_return": training["final_100_return"], "final_100_loss": training["final_100_loss"],
                                    "training_wall_seconds": training["wall_seconds"],
                                    "metrics_row_matches": seed_rows[(method, seed)]["training"] == training})
    checks["checkpoints_hash_identity_and_config"] = all(
        r["sha256"] == r["inventory_sha256"] and r["method"] in r["path"] and str(r["seed"]) in r["path"] and r["parameters"] == r["recorded_parameters"]
        and r["candidate_matches_freeze"] and r["episodes"] == freeze["episodes_per_seed"] and r["metrics_row_matches"] for r in checkpoint_rows)
    all_checkpoints = sorted(p.name for p in (bundle / "checkpoints").glob("*"))
    checks["exactly_one_final_checkpoint_per_model"] = all_checkpoints == sorted(Path(r["path"]).name for r in checkpoint_rows)
    (output / "checkpoint_audit.json").write_text(json.dumps({"checkpoints": checkpoint_rows, "files_present": all_checkpoints}, indent=2) + "\n")
    emit({"kind": "checkpoint_audit", "checkpoints": len(checkpoint_rows), "ok": checks["checkpoints_hash_identity_and_config"]})

    # 6. Budget and matching safety -------------------------------------------
    budgets = {c: catalog_data[c]["bell_pair_budget"] for c in catalogs}
    safety = {"learned_rows": len(pair_rows),
              "attempted_plus_unused_equals_budget": all(a + u == budgets[c] for c, a, u, _ in pair_rows),
              "attempted_within_budget": all(0 <= a <= budgets[c] for c, a, _, _ in pair_rows),
              "unused_nonnegative": all(u >= 0 for _, _, u, _ in pair_rows),
              "surplus_nonnegative": all(s >= -1e-12 for *_, s in pair_rows),
              "recorded_violations": recorded["violations"],
              "mean_pairs_unused_by_catalog": {c: float(np.mean([u for cc, _, u, _ in pair_rows if cc == c])) for c in catalogs},
              "maximum_simultaneous_pairs": {c: catalog_data[c].get("maximum_simultaneous_pairs") for c in catalogs},
              "matching_evidence": ("Per-step matchings are not stored in raw rows. Feasibility is enforced by the evaluator, which raises on "
                                    "any infeasible or endpoint-sharing action; the run completed with an empty violation list, so every "
                                    "evaluated action was a feasible matching. This audit cannot independently replay matchings without "
                                    "allocator decisions, which the Stage 29 cap forbids.")}
    unused = defaultdict(list)
    for c, a, u, s in pair_rows:
        unused[c].append(u)
    checks["budget_and_matching_safety"] = (safety["attempted_plus_unused_equals_budget"] and safety["attempted_within_budget"]
                                           and safety["unused_nonnegative"] and safety["surplus_nonnegative"] and not recorded["violations"])
    (output / "safety_audit.json").write_text(json.dumps(safety, indent=2) + "\n")
    emit({"kind": "safety_audit", **{k: v for k, v in safety.items() if isinstance(v, bool)}})

    # 7. Data-split disjointness ----------------------------------------------
    splits = {name: json.loads((ROOT / f"configs/data/qip_v1/{name}-request-types.json").read_text()) for name in ("development", "validation", "test")}
    ids = {name: {r["type_id"] for r in data["types"]} for name, data in splits.items()}
    contents = {name: {json.dumps([r["predicate"], r["weights"]]) for r in data["types"]} for name, data in splits.items()}
    pairs = list(itertools.combinations(ids, 2))
    split_lock = json.loads((ROOT / "configs/data/qip_v1/split-lock.json").read_text())
    lock_ok = True
    lock_rows = []

    def walk(node):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    for item in walk(split_lock["splits"]):
        ok = _sha(ROOT / item["path"]) == item["sha256"]; lock_ok &= ok; lock_rows.append({"path": item["path"], "match": ok})
    lock_ok &= len(lock_rows) >= 19
    catalog_split_ok = True
    for path in sorted((ROOT / "configs/data/qip_v1").glob("*-catalog.json")):
        data = json.loads(path.read_text()); split = path.name.split("-")[0]
        used = {r["type_id"] for s in data["states"] for r in s["requests"]}
        catalog_split_ok &= used <= ids[split]
    training_ids = {ref["catalog_id"] for refs in freeze["training_catalogs"].values() for ref in refs}
    # Public-dynamics overlap: identical transition matrices / active-edge-and-urgency patterns across catalogs.
    all_catalogs = {json.loads(p.read_text())["catalog_id"]: json.loads(p.read_text()) for p in sorted((ROOT / "configs/data/qip_v1").glob("*-catalog.json"))}
    dynamics_overlap = []
    names = sorted(all_catalogs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = all_catalogs[a], all_catalogs[b]
            same_transition = (np.shape(A["transition"]) == np.shape(B["transition"])
                               and float(np.max(np.abs(np.asarray(A["transition"]) - np.asarray(B["transition"])))) == 0.0)
            pattern = lambda c: [[(r["edge_index"], r["urgency"]) for r in s["requests"]] for s in c["states"]]
            same_pattern = pattern(A) == pattern(B)
            if same_transition or same_pattern:
                dynamics_overlap.append({"catalog_a": a, "catalog_b": b, "generator_seed_a": A["generator_seed"], "generator_seed_b": B["generator_seed"],
                                         "identical_transition_matrix": same_transition, "identical_active_edge_and_urgency_pattern": same_pattern,
                                         "same_edge_list": A["edges"] == B["edges"], "horizon": [A["horizon"], B["horizon"]],
                                         "budget": [A["bell_pair_budget"], B["bell_pair_budget"]]})
    test_overlap = [row for row in dynamics_overlap if "test" in row["catalog_a"] + row["catalog_b"]
                    and any(row[k] in training_ids for k in ("catalog_a", "catalog_b"))]
    fully_fresh_test = [c for c in catalogs if not any(c in (row["catalog_a"], row["catalog_b"]) for row in test_overlap)]
    split_audit = {"request_type_counts": {k: len(v) for k, v in ids.items()},
                   "type_id_overlap": {f"{a}-{b}": len(ids[a] & ids[b]) for a, b in pairs},
                   "content_overlap": {f"{a}-{b}": len(contents[a] & contents[b]) for a, b in pairs},
                   "catalogs_use_only_own_split_types": catalog_split_ok,
                   "training_catalogs_exclude_test": not any("test" in c for c in training_ids),
                   "test_catalogs_are_test_split": all(c.startswith("qip-test-") for c in catalogs),
                   "split_lock_entries": lock_rows, "split_lock_hashes_match": lock_ok,
                   "training_catalog_ids": sorted(training_ids), "test_catalog_ids": catalogs,
                   "public_dynamics_overlap": dynamics_overlap, "test_catalogs_sharing_training_dynamics": test_overlap,
                   "test_catalogs_with_fresh_dynamics": fully_fresh_test,
                   "cause": ("scripts/generate_qip_splits.py seeds catalogs with split_seed + 100 + offset, so development, validation, "
                             "and test seeds 25202-25208 collide; catalogs with equal seeds and equal edge counts draw identical "
                             "active-edge patterns, urgencies, and transition matrices. Request types are drawn from disjoint split-specific lists.")}
    if test_overlap:
        findings.append({"severity": "material", "check": "test_public_dynamics_overlap",
                         "detail": "Sealed test catalogs " + ", ".join(sorted({r["catalog_a"] if "test" in r["catalog_a"] else r["catalog_b"] for r in test_overlap}))
                                   + " reuse the public Markov dynamics (transition matrix, active-edge pattern, urgencies) of training catalogs; "
                                   "only their request types are fresh. Fully fresh test catalogs: " + ", ".join(fully_fresh_test) + "."})
    if any("validation" in r["catalog_a"] + r["catalog_b"] and "development" in r["catalog_a"] + r["catalog_b"] for r in dynamics_overlap):
        findings.append({"severity": "material", "check": "validation_public_dynamics_overlap",
                         "detail": "A validation catalog reuses a development catalog's public dynamics; Stage 26-27 validation was not fully independent for that catalog."})
    checks["request_type_splits_disjoint"] = (all(v == 0 for v in split_audit["type_id_overlap"].values())
                                      and all(v == 0 for v in split_audit["content_overlap"].values())
                                      and catalog_split_ok and split_audit["training_catalogs_exclude_test"] and lock_ok)
    (output / "split_audit.json").write_text(json.dumps(split_audit, indent=2) + "\n")
    emit({"kind": "split_audit", "ok": checks["request_type_splits_disjoint"], "test_catalogs_sharing_training_dynamics": len(test_overlap),
          "test_catalogs_with_fresh_dynamics": fully_fresh_test})

    # Post hoc, clearly non-registered descriptive sensitivity: primary contrasts restricted to the fully fresh catalogs.
    if fully_fresh_test:
        fresh_ppo = np.asarray([np.mean([seed_catalog["ppo"][s][c] for c in fully_fresh_test]) for s in seeds])
        fresh_dqn = np.asarray([np.mean([seed_catalog["dqn"][s][c] for c in fully_fresh_test]) for s in seeds])
        fresh_greedy = float(np.mean([baseline_table["greedy"][c] for c in fully_fresh_test]))
        fresh_online = float(np.mean([baseline_table["online_reference"][c] for c in fully_fresh_test]))
        posthoc = {"status": "post hoc descriptive sensitivity; not a registered test; no decision rule",
                   "catalogs": fully_fresh_test, "ppo_mean": float(fresh_ppo.mean()), "dqn_mean": float(fresh_dqn.mean()),
                   "greedy_mean": fresh_greedy, "online_reference_mean": fresh_online,
                   "ppo_minus_dqn": float((fresh_ppo - fresh_dqn).mean()),
                   "ppo_minus_dqn_95": percentile_bootstrap(fresh_ppo - fresh_dqn, 29_100, reps, 2.5),
                   "ppo_minus_dqn_90": percentile_bootstrap(fresh_ppo - fresh_dqn, 29_101, reps, 5.0),
                   "ppo_minus_greedy": float(fresh_ppo.mean() - fresh_greedy),
                   "ppo_minus_greedy_95": percentile_bootstrap(fresh_ppo - fresh_greedy, 29_102, reps, 2.5),
                   "dqn_minus_greedy_95": percentile_bootstrap(fresh_dqn - fresh_greedy, 29_103, reps, 2.5),
                   "five_percent_margin": 0.05 * fresh_online,
                   "ppo_fraction_online": float(fresh_ppo.mean() / fresh_online), "dqn_fraction_online": float(fresh_dqn.mean() / fresh_online),
                   "greedy_fraction_online": fresh_greedy / fresh_online}
        reproduced["posthoc_fresh_dynamics_sensitivity"] = posthoc
        (output / "reproduced_analysis.json").write_text(json.dumps(reproduced, indent=2) + "\n")
        emit({"kind": "posthoc_sensitivity", **{k: v for k, v in posthoc.items() if k != "status"}})

    # 8. Test-lock compliance ---------------------------------------------------
    sealed = [freeze["test_request_manifest"]["path"], *[ref["path"] for ref in freeze["test_catalogs"]]]
    lock_commit = config["test_lock_commit"]
    unchanged = {path: _git("rev-parse", f"{lock_commit}:{path}") == _git("rev-parse", f"HEAD:{path}") for path in sealed}
    touched = {path: _git("log", "--format=%H", f"{lock_commit}..HEAD", "--", path).split() for path in sealed}
    earlier_leak = {}
    for relative in config["pre_test_bundles"]:
        # The Stage 27 freeze names (path, hash) the sealed files by design; every other record must not.
        files = [p for p in (ROOT / relative).glob("*.json*") if p.name != "stage28-freeze.json"]
        earlier_leak[relative] = [p.name for p in files if "qip-test-" in p.read_text(errors="ignore")]
    freeze_before_run = subprocess.run(["git", "merge-base", "--is-ancestor", config["freeze_commit"], config["stage28_source_commit"]],
                                       cwd=ROOT).returncode == 0
    freeze_in_commit = _git("rev-parse", f"{config['freeze_commit']}:{config['freeze']['path']}") == _git("rev-parse", f"HEAD:{config['freeze']['path']}")
    attempts = json.loads((ROOT / "experiments/28/index.json").read_text())["runs"]
    test_lock = {"sealed_files_unchanged_since_stage25_lock": unchanged, "commits_touching_sealed_files_after_lock": touched,
                 "test_identifiers_in_pre_stage28_bundles": earlier_leak, "freeze_commit_is_ancestor_of_stage28_source": freeze_before_run,
                 "freeze_file_unchanged_since_freeze_commit": freeze_in_commit,
                 "stage28_attempts": [a["run_id"] for a in attempts], "final_checkpoints_only": checks["exactly_one_final_checkpoint_per_model"],
                 "post_test_changes": "None. Stage 29 performs no training, tuning, selection, seed removal, or budget extension."}
    checks["test_lock_compliance"] = (all(unchanged.values()) and not any(touched.values()) and not any(earlier_leak.values())
                                      and freeze_before_run and freeze_in_commit and len(attempts) == 1)
    (output / "test_lock_audit.json").write_text(json.dumps(test_lock, indent=2) + "\n")
    emit({"kind": "test_lock_audit", "ok": checks["test_lock_compliance"]})

    # 9. Resource accounting ----------------------------------------------------
    horizons = {c: catalog_data[c]["horizon"] for c in catalogs}
    training_decisions = sum(r["decisions"] for r in checkpoint_rows)
    allocator_updates = sum(r["updates"] for r in checkpoint_rows)
    primary_eval = 2 * len(seeds) * sum(len(primary) * horizons[c] for c in catalogs)
    secondary_eval = len(seeds) * len(noisy_conditions) * sum(len(secondary) * horizons[c] for c in catalogs)
    baseline_eval = 4 * sum(len(primary) * horizons[c] for c in catalogs)
    certification_updates = len(test_manifest["types"]) * 16 * 800
    reported = manifest["resources_consumed"]
    resources = {"allocator_training_decisions": training_decisions, "learned_evaluation_decisions_primary": primary_eval,
                 "learned_evaluation_decisions_secondary": secondary_eval, "baseline_policy_decisions": baseline_eval,
                 "total_allocator_decisions": training_decisions + primary_eval + secondary_eval + baseline_eval,
                 "allocator_optimizer_updates": allocator_updates, "certification_optimizer_updates": certification_updates,
                 "certification_circuit_evaluations": certification_updates * 9, "shots": 0,
                 "wall_seconds": reported["wall_seconds"], "cpu_core_hours": reported["cpu_core_hours"], "gpu_device_hours": reported["gpu_device_hours"],
                 "training_wall_seconds_by_method": {m: float(sum(r["training_wall_seconds"] for r in checkpoint_rows if r["method"] == m)) for m in ("dqn", "ppo")},
                 "parameters_by_method": {m: next(r["parameters"] for r in checkpoint_rows if r["method"] == m) for m in ("dqn", "ppo")},
                 "caps": {"wall_seconds": 14400, "decisions": 2_500_000, "max_cpu_threads": 4},
                 "recorded_job_threads": manifest["runtime"]["hardware"]["max_job_threads"], "reported": reported,
                 "notes": ["Decisions exclude the online-reference and future-clairvoyant dynamic programs and the message-oracle arithmetic.",
                           "Circuit evaluations are analytic float64 Bell-correlator evaluations during request certification (9 per optimizer update); no shots were sampled and no hardware was used.",
                           "The reported 'updates' counter sums allocator and certification optimizer updates; this audit reports them separately.",
                           "CPU core-hours are process user+system CPU time."]}
    checks["resource_accounting_reconciled"] = (resources["total_allocator_decisions"] == reported["decisions"]
                                               and allocator_updates + certification_updates == reported["updates"]
                                               and resources["certification_circuit_evaluations"] == reported["circuit_evaluations"]
                                               and reported["shots"] == 0)
    checks["within_registered_caps"] = (reported["decisions"] <= 2_500_000 and reported["wall_seconds"] <= 14400
                                       and resources["recorded_job_threads"] <= 4)
    program = {}
    for relative in config["program_bundles"]:
        m = json.loads((ROOT / relative / "manifest.json").read_text())
        program[m["run_id"]] = {k: m["resources_consumed"][k] for k in ("decisions", "updates", "wall_seconds", "cpu_core_hours", "gpu_device_hours", "circuit_evaluations", "shots")}
    resources["qip_program_runs"] = program
    resources["qip_program_totals"] = {k: float(sum(v[k] for v in program.values())) for k in next(iter(program.values()))}
    (output / "resource_audit.json").write_text(json.dumps(resources, indent=2) + "\n")
    emit({"kind": "resource_audit", "decisions": resources["total_allocator_decisions"], "ok": checks["resource_accounting_reconciled"]})

    # 10. Earlier-stage claims re-derived from raw records ---------------------
    s23 = ROOT / config["stage23_bundle"]
    s23_rows = [r for r in _rows(s23 / "metrics.jsonl") if r["kind"] == "confirmatory_seed"]
    s23_base = defaultdict(list)
    for r in _rows(s23 / "episode_results.jsonl"):
        if r["kind"] == "baseline_episode":
            s23_base[r["method"]].append(r["surplus"])
    s23_means = {m: float(np.mean(v)) for m, v in s23_base.items()}
    s23_surplus = np.asarray([r["mean_surplus"] for r in s23_rows]); s23_fraction = np.asarray([r["dp_fraction"] for r in s23_rows])
    s23_recorded = json.loads((s23 / "confirmatory_analysis.json").read_text())
    stage23 = {"ber_marl_mean": float(s23_surplus.mean()), "greedy": s23_means["quantum_greedy"], "random": s23_means["quantum_random"],
               "static": s23_means["quantum_static"], "online_dp": s23_means["dp_online_oracle"], "future_clairvoyant": s23_means["future_clairvoyant"],
               "fraction_of_online_dp_mean": float(s23_fraction.mean()),
               "fraction_of_online_dp_overall": float(s23_surplus.mean() / s23_means["dp_online_oracle"]),
               "fraction_one_sided_95_lower": percentile_bootstrap(s23_fraction, 24000, 20000, 5.0)[0],
               "difference_vs_greedy_95": percentile_bootstrap(s23_surplus - s23_means["quantum_greedy"], 23999, 20000, 2.5),
               "seeds": len(s23_rows), "seeds_above_greedy": int((s23_surplus > s23_means["quantum_greedy"]).sum()),
               "gates": s23_recorded["gates"]}
    s26 = json.loads((ROOT / config["stage26_recovery"] / "recovered_selection.json").read_text())
    s26_rows = list(_rows(ROOT / config["stage26_attempt"] / "metrics.jsonl"))
    s26_methods = {}
    for method in ("dqn", "double_dqn", "ppo", "recurrent_ppo"):
        scores = defaultdict(list)
        for r in s26_rows:
            if r["kind"] == "tuning_trial" and r["method"] == method:
                scores[r["candidate"]].append(r["validation_score"])
        best = max(scores, key=lambda k: np.mean(scores[k]))
        s26_methods[method] = {"selected": best, "score": float(np.mean(scores[best])),
                               "all_candidates": {k: float(np.mean(v)) for k, v in scores.items()}}
    s26_ablation = defaultdict(list)
    for r in s26_rows:
        if r["kind"] == "ablation":
            s26_ablation[r["feature_mode"]].append(r["validation_score"])
    stage26 = {"methods": s26_methods, "ablations": {k: float(np.mean(v)) for k, v in s26_ablation.items()},
               "double_dqn_minus_ppo": s26_methods["double_dqn"]["score"] - s26_methods["ppo"]["score"],
               "recorded": {"method_scores": s26["method_scores"], "ablation_scores": s26["ablation_scores"]}}
    stage26["reproduced"] = (all(abs(stage26["methods"][m]["score"] - s26["method_scores"][m]) < TOLERANCE for m in s26_methods)
                             and all(abs(stage26["ablations"][k] - s26["ablation_scores"][k]) < TOLERANCE for k in s26["ablation_scores"]))
    s27_rows = list(_rows(ROOT / config["stage27_bundle"] / "metrics.jsonl"))
    s27 = defaultdict(lambda: defaultdict(list)); s27_cat = defaultdict(lambda: defaultdict(lambda: defaultdict(list))); s27_ref = defaultdict(dict)
    s27_meta = {}
    for r in s27_rows:
        if r["kind"] == "robustness_evaluation":
            s27_cat[r["method"]][r["condition"]][r["catalog_id"]].append(r["mean_surplus"])
            s27_meta[r["catalog_id"]] = {"agents": r["agents"], "topology": r["topology"], "horizon": r["horizon"], "budget": r["budget"]}
        elif r["kind"] == "robustness_reference":
            s27_ref[r["condition"]][r["catalog_id"]] = {"greedy": r["greedy_mean"], "online_reference": r["online_reference_mean"]}
    s27_means = {m: {cond: float(np.mean([np.mean(v) for v in cats.values()])) for cond, cats in conds.items()} for m, conds in s27_cat.items()}
    s27_recorded = json.loads((ROOT / config["stage27_bundle"] / "robustness_analysis.json").read_text())
    stage27 = {"means": s27_means,
               "per_catalog": {m: {cond: {c: float(np.mean(v)) for c, v in cats.items()} for cond, cats in conds.items()} for m, conds in s27_cat.items()},
               "references": s27_ref, "catalog_metadata": s27_meta, "positive_gain_fractions": s27_recorded["positive_gain_fractions"]}
    stage27["reproduced"] = all(abs(s27_means[m][cond] - s27_recorded["aggregate"][m][cond]["mean"]) < TOLERANCE
                                for m in s27_means for cond in s27_means[m])
    non_test = json.loads((ROOT / freeze["non_test_certificate_bundle"]["path"]).read_text())["certificates"]
    nt_threshold = []
    for row in non_test:
        denom = row["bell_value"] - 0.5
        nt_threshold.append(min(1.0, (row["classical_value"] - 0.5) / denom) if denom > 0 else 1.0)
    stage27["non_test_visibility_thresholds"] = {"min": float(np.min(nt_threshold)), "median": float(np.median(nt_threshold)), "max": float(np.max(nt_threshold)), "count": len(nt_threshold)}
    checks["earlier_stage_numbers_reproduced"] = stage26["reproduced"] and stage27["reproduced"] and stage23["seeds"] == 16

    claims = {
        "supported": [
            {"claim": "Stage 23: learned allocator beats myopic greedy in the four-agent budgeted benchmark", "evidence": stage23["difference_vs_greedy_95"], "seeds_above_greedy": stage23["seeds_above_greedy"]},
            {"claim": "Stage 28: PPO beats greedy on sealed catalogs (registered primary test 2)", "evidence": greedy95},
            {"claim": "Stage 28: PPO and DQN practically equivalent within 5% of online-reference surplus (registered primary test 3)", "evidence": diff90, "margin": margin},
            {"claim": "Stage 26: removing budget, horizon, gain magnitude, or gain location reduces validation surplus", "evidence": stage26["ablations"]},
            {"claim": "Stages 27-28: heralded generation loss degrades surplus gradually", "evidence": {"stage27_ppo": {k: s27_means["ppo"][k] for k in ("ideal", "heralded-0.9", "heralded-0.75", "heralded-0.5")}, "stage28_ppo": {"ideal_first_100": ideal_first100, **{k: secondary_means[k] for k in ("heralded-0.75", "heralded-0.5")}}}},
        ],
        "null_or_failed": [
            {"claim": "Stage 23: one-sided 95% lower bound of fraction of online DP surplus exceeds 0.90", "outcome": "failed", "evidence": stage23["fraction_one_sided_95_lower"]},
            {"claim": "Stage 26: registered Double DQN improves over the strongest learned comparator", "outcome": "null", "evidence": stage26["double_dqn_minus_ppo"]},
            {"claim": "Stage 27: broad robustness to Werner noise", "outcome": "failed", "evidence": {m: {k: s27_means[m][k] for k in ("visibility-0.9", "visibility-0.8")} for m in s27_means}},
            {"claim": "Stage 28: PPO superior to DQN (registered primary test 1)", "outcome": "null", "evidence": diff95},
            {"claim": "Stage 28: per-catalog PPO>DQN after Holm adjustment", "outcome": "null on all four catalogs", "evidence": [r["holm_p"] for r in catalog_tests]},
            {"claim": "Stage 28: surplus under Werner visibility 0.8 and 0.6", "outcome": "zero (collapse)", "evidence": {k: secondary_means[k] for k in ("visibility-0.8", "visibility-0.6")}},
        ],
        "qualifications": [f for f in findings if f["severity"] == "material"],
        "not_claimed": ["new Bell violation", "quantum computational speedup", "hardware evidence", "decentralized allocator",
                        "general quantum advantage beyond the certified XOR request family and tested resource models",
                        "PPO superiority over DQN", "near-oracle (90%) performance"],
    }
    (output / "claims_audit.json").write_text(json.dumps(claims, indent=2) + "\n")
    earlier = {"stage23": stage23, "stage26": stage26, "stage27": stage27}
    (output / "earlier_stage_audit.json").write_text(json.dumps(earlier, indent=2) + "\n")
    emit({"kind": "earlier_stage_audit", "stage26_reproduced": stage26["reproduced"], "stage27_reproduced": stage27["reproduced"]})

    # 11. Paper tables and figure source data ---------------------------------
    figure_dir = output / "figure_data"; figure_dir.mkdir(exist_ok=True)
    labels = {"qip-test-id-a4-complete": "4-agent complete (fresh requests)", "qip-test-topology-a6-ring": "6-agent ring",
              "qip-test-scale-a8-sparse": "8-agent sparse", "qip-test-horizon-a4-path": "4-agent path, horizon 24"}

    def write_csv(name, header, rows):
        with (figure_dir / name).open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)

    order = ["random", "static", "greedy", "dqn", "ppo", "online_reference", "future_clairvoyant"]
    write_csv("fig_main_by_catalog.csv", ["catalog_id", "label", "method", "mean_surplus", "seed_sd", "fraction_of_online_reference"],
              [[c, labels[c], m,
                (float(np.mean([seed_catalog[m][s][c] for s in seeds])) if m in ("dqn", "ppo") else baseline_table[m][c]),
                (float(np.std([seed_catalog[m][s][c] for s in seeds], ddof=1)) if m in ("dqn", "ppo") else 0.0),
                (float(np.mean([seed_catalog[m][s][c] for s in seeds])) if m in ("dqn", "ppo") else baseline_table[m][c]) / baseline_table["online_reference"][c]]
               for c in catalogs for m in order])
    write_csv("fig_seed_catalog.csv", ["catalog_id", "method", "training_seed", "mean_surplus"],
              [[c, m, s, seed_catalog[m][s][c]] for c in catalogs for m in ("dqn", "ppo") for s in seeds])
    write_csv("fig_seed_contrasts.csv", ["training_seed", "ppo_mean", "dqn_mean", "ppo_minus_dqn", "ppo_minus_greedy"],
              [[s, float(seed_mean["ppo"][i]), float(seed_mean["dqn"][i]), float(difference[i]), float(versus_greedy[i])] for i, s in enumerate(seeds)])
    write_csv("fig_primary_intervals.csv", ["test", "estimate", "lower", "upper", "interval", "threshold_low", "threshold_high", "decision"],
              [["ppo_minus_dqn_superiority", float(difference.mean()), *diff95, "two-sided 95% (lower bound used)", 0.0, "", tests["ppo_superior_to_dqn"]],
               ["ppo_minus_dqn_equivalence", float(difference.mean()), *diff90, "two-sided 90%", -margin, margin, tests["ppo_dqn_equivalent_within_5pct_online_reference"]],
               ["ppo_minus_greedy_superiority", float(versus_greedy.mean()), *greedy95, "two-sided 95% (lower bound used)", 0.0, "", tests["ppo_superior_to_greedy"]]])
    write_csv("fig_noise_stage28.csv", ["condition", "visibility", "heralded_success", "ppo_mean_surplus", "seed_sd", "sequences_per_catalog"],
              [["ideal", 1.0, 1.0, ideal_first100, float(np.std([np.mean([np.mean([learned[("ppo", "ideal", c, s)][q] for q in secondary]) for c in catalogs]) for s in seeds], ddof=1)), len(secondary)]]
              + [[cond, {"visibility-0.8": 0.8, "visibility-0.6": 0.6}.get(cond, 1.0), {"heralded-0.75": 0.75, "heralded-0.5": 0.5}.get(cond, 1.0),
                  secondary_means[cond], float(np.std(list(secondary_seed[cond].values()), ddof=1)), len(secondary)] for cond in noisy_conditions])
    write_csv("fig_noise_stage27.csv", ["method", "condition", "mean_surplus"],
              [[m, cond, v] for m, d in s27_means.items() for cond, v in d.items()])
    write_csv("fig_visibility_thresholds.csv", ["split", "type_id", "classical_value", "bell_value", "visibility_threshold"],
              [["test", r["type_id"], r["classical_exact"], r["bell_at_recorded_angles"], r["visibility_threshold"]] for r in certificate_rows]
              + [[row["split"], row["type_id"], row["classical_value"], row["bell_value"], t] for row, t in zip(non_test, nt_threshold)])
    write_csv("fig_positive_fraction_test.csv", ["visibility", "fraction_of_test_opportunities_with_positive_gain"],
              [[float(k), v] for k, v in positive_fraction.items()])
    write_csv("fig_ablation_stage26.csv", ["condition", "mean_validation_surplus"],
              [["full", stage26["methods"]["double_dqn"]["score"]]] + [[k, v] for k, v in sorted(stage26["ablations"].items())])
    write_csv("fig_stage26_methods.csv", ["method", "selected_config", "mean_validation_surplus"],
              [[m, d["selected"], d["score"]] for m, d in stage26["methods"].items()])
    scaling_rows = []
    for c, meta in sorted(s27_meta.items(), key=lambda kv: (kv[1]["agents"], kv[0])):
        scaling_rows.append(["stage27-validation", c, meta["agents"], meta["topology"], meta["horizon"], meta["budget"],
                             float(np.mean(s27_cat["ppo"]["ideal"][c])), float(np.mean(s27_cat["dqn"]["ideal"][c])),
                             s27_ref["ideal"][c]["greedy"], s27_ref["ideal"][c]["online_reference"]])
    for c in catalogs:
        d = catalog_data[c]
        scaling_rows.append(["stage28-test", c, d["agents"], d.get("topology"), d["horizon"], d["bell_pair_budget"],
                             float(np.mean([seed_catalog["ppo"][s][c] for s in seeds])), float(np.mean([seed_catalog["dqn"][s][c] for s in seeds])),
                             baseline_table["greedy"][c], baseline_table["online_reference"][c]])
    write_csv("fig_scaling.csv", ["split", "catalog_id", "agents", "topology", "horizon", "budget", "ppo", "dqn", "greedy", "online_reference"], scaling_rows)
    write_csv("fig_training_curves_summary.csv", ["method", "training_seed", "final_100_return", "final_100_loss", "training_wall_seconds", "updates"],
              [[r["method"], r["seed"], r["final_100_return"], r["final_100_loss"], r["training_wall_seconds"], r["updates"]] for r in checkpoint_rows])
    figure_hashes = {p.name: _sha(p) for p in sorted(figure_dir.glob("*.csv"))}
    paper_tables = {"table_main": {"method_means": method_means, "by_catalog": {c: {**{m: baseline_table[m][c] for m in baseline_methods},
                                                                                   "ppo": float(np.mean([seed_catalog["ppo"][s][c] for s in seeds])),
                                                                                   "dqn": float(np.mean([seed_catalog["dqn"][s][c] for s in seeds]))} for c in catalogs}},
                    "table_primary_tests": {"ppo_minus_dqn": float(difference.mean()), "ppo_minus_dqn_95": diff95, "ppo_minus_dqn_90": diff90,
                                            "ppo_minus_greedy": float(versus_greedy.mean()), "ppo_minus_greedy_95": greedy95,
                                            "equivalence_margin": margin, "decisions": tests},
                    "table_catalog_tests": catalog_tests, "table_noise": {"stage28": {"ideal_first_100": ideal_first100, **secondary_means}, "stage27": s27_means},
                    "table_resources": resources, "table_stage26": stage26, "table_stage23": stage23,
                    "catalog_labels": labels, "catalog_metadata": {c: {k: catalog_data[c].get(k) for k in ("agents", "topology", "horizon", "bell_pair_budget")}
                                                                   | {"edges": len(catalog_data[c]["edges"]), "states": len(catalog_data[c]["states"])} for c in catalogs},
                    "figure_data_sha256": figure_hashes}
    (output / "paper_tables.json").write_text(json.dumps(paper_tables, indent=2) + "\n")

    gates = {**checks}
    failed = [k for k, v in gates.items() if not v]
    for name in failed:
        findings.append({"severity": "blocking", "check": name})
    findings.extend(config.get("recorded_findings", []))
    emit({"kind": "audit_gates", **gates})
    return {"passed": not failed, "gates": gates, "findings": findings,
            "resources": {"primitive_steps": 0, "decisions": 0, "updates": 0, "circuit_evaluations": len(certificate_rows), "shots": 0},
            "summary": {"scientific_tests": tests, "method_means": method_means, "visibility_thresholds": threshold_summary}}
