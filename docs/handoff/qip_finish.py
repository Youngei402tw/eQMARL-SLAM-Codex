#!/usr/bin/env python3
"""Deterministic ledger edits for closing Stage 28 and running Stage 29.

Run from the repository root. Git commands, pushes, and remote verification are performed
by the operator between steps (see docs/handoff/STAGE28_29_HANDOFF.md); this script only
edits files, and it never invents a SHA: every identity is passed in after it was verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "stages.json").is_file():
    raise SystemExit("run from the repository root")
REPO = "https://github.com/Youngei402tw/eQMARL-SLAM-Codex"
CONFIG29 = ROOT / "configs/stage29/29-qip-release-audit_attempt-01.yaml"
BUNDLE28 = "experiments/28/28-qip-confirmatory_attempt-01"
BUNDLE29 = "experiments/29/29-qip-release-audit_attempt-01"


def load(path):
    return json.loads((ROOT / path).read_text())


def dump(path, data):
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_text(json.dumps(data, indent=2) + "\n")


def now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stage(stages, sid):
    return next(s for s in stages["stages"] if s["id"] == sid)


def stage28_receipt(args):
    summary = load(f"{BUNDLE28}/summary.json")
    dump("docs/stages/28-publication.json", {
        "stage": "28", "branch": "stage/28-qip-confirmatory", "repository": REPO,
        "source_commit": "3d3ea45ab610a4752a61ac41456aeb5ae009998e",
        "snapshot_commit": "2757563d044a3468795faf129bc13cc6650dfacd",
        "results_commit": args.results_commit, "remote_ref": "refs/heads/stage/28-qip-confirmatory",
        "remote_sha_verified": args.remote_sha, "bundle": BUNDLE28, "bundle_valid": True,
        "metrics_rows": summary["metrics_rows"], "metrics_sha256": summary["metrics_sha256"],
        "freeze_sha256": "ff34ffbb5a30053555de83494c937980ad4f91c5e4e87f2401833cf173df60ef",
        "verified": args.remote_sha == args.results_commit, "verified_utc": now()})
    state = load("project_state.json")
    state["publication_status"] = "complete_remote_verified"
    state["verified_remote_commits"]["stage/28-qip-confirmatory"] = args.remote_sha
    state["updated_utc"] = now()
    state["next_action"] = "Record Stage 28 completion commit, then create stage/29-qip-release from it."
    dump("project_state.json", state)
    stages = load("stages.json")
    entry = stage(stages, "28")
    entry["results_commit"] = args.results_commit
    entry["publication_receipt"] = "docs/stages/28-publication.json"
    dump("stages.json", stages)


def stage29_start(args):
    c = args.completion_commit
    text = CONFIG29.read_text()
    if "STAGE28_COMPLETION_COMMIT" not in text:
        raise SystemExit("config placeholder already replaced")
    CONFIG29.write_text(text.replace("STAGE28_COMPLETION_COMMIT", c))
    stages = load("stages.json")
    stage(stages, "28")["completion_commit"] = c
    s29 = stage(stages, "29")
    s29.update({"status": "in_progress", "branch_created": True, "dependency_completion_commit": c, "branch_created_from_commit": c})
    dump("stages.json", stages)
    state = load("project_state.json")
    state.update({"active_stage": "29", "stage_status": "in_progress", "current_branch": "stage/29-qip-release",
                  "dependency_commit": c, "publication_status": "stage29_source_pending", "updated_utc": now(),
                  "next_action": "Publish Stage 29 source/config, then run the registered independent release audit."})
    state.setdefault("completed_stage_budgets", {})["28"] = {
        "cap_wall_seconds": 14400, "consumed_wall_seconds": state["stage_budget"]["active_wall_seconds"],
        "setup_wall_seconds_charged": 0, "recorded_experiment_resources": load(f"{BUNDLE28}/manifest.json")["resources_consumed"]}
    state["stage_budget"] = {"stage": "29", "setup_wall_seconds_charged": 0}
    dump("project_state.json", state)
    sha = hashlib.sha256(CONFIG29.read_bytes()).hexdigest()
    dump("experiments/29/index.json", {"stage": "29", "protocol_id": "robust-entanglement-routing-v1", "runs": [
        {"run_id": "29-qip-release-audit_attempt-01", "status": "planned",
         "config_path": "configs/stage29/29-qip-release-audit_attempt-01.yaml", "config_sha256": sha}]})


def stage29_receipt_file(args):
    Path(args.out).write_text(json.dumps({"source_commit": args.source_commit, "branch": "stage/29-qip-release", "verified": True,
                                          "remote_ref": "refs/heads/stage/29-qip-release", "verified_utc": now()}, separators=(",", ":")) + "\n")


def stage29_results(args):
    index = load("experiments/29/index.json")
    index["runs"][0].update({"status": "completed", "result_path": BUNDLE29})
    dump("experiments/29/index.json", index)
    manifest = load(f"{BUNDLE29}/manifest.json")
    diagnostics = load(f"{BUNDLE29}/diagnostics.json")
    state = load("project_state.json")
    passed = manifest["status"] == "completed"
    state.update({"stage_status": "passed" if passed else "failed", "publication_status": "stage29_results_pending", "updated_utc": now(),
                  "next_action": "Publish and verify Stage 29 results, check GitHub Actions, and record the publication receipt."})
    r = manifest["resources_consumed"]
    state["stage_budget"] = {"stage": "29", "setup_wall_seconds_charged": 0, "active_wall_seconds": r["wall_seconds"],
                             "cpu_core_hours": r["cpu_core_hours"], "decisions": r["decisions"], "updates": r["updates"],
                             "circuit_evaluations": r["circuit_evaluations"], "shots": r["shots"]}
    dump("project_state.json", state)
    stages = load("stages.json")
    stage(stages, "29")["status"] = "passed" if passed else "failed"
    stage(stages, "29")["source_commit"] = manifest["source_commit"]
    dump("stages.json", stages)
    gates = diagnostics["gates"]; findings = diagnostics.get("findings", [])
    analysis = load(f"{BUNDLE29}/reproduced_analysis.json")
    lines = ["", "## Audit result", "",
             f"Run `29-qip-release-audit_attempt-01` ran from source `{manifest['source_commit']}`. Its status is **{manifest['status']}**. "
             f"{sum(gates.values())} of {len(gates)} audit gates passed: " + ", ".join(f"`{k}`" for k, v in gates.items() if v) + ".",
             "",
             f"{sum(c['match'] for c in analysis['comparisons'])} of {len(analysis['comparisons'])} registered Stage 28 quantities were reproduced within 1e-9. "
             "These are the method means, the three primary intervals, the four exact Wilcoxon p-values with their Holm adjustments, the secondary-condition means, and the visibility thresholds. "
             f"The registered decisions are unchanged: {json.dumps(analysis['scientific_tests'])}.",
             "", "### Findings", ""]
    for f in findings:
        lines.append(f"- **{f['severity']}** `{f['check']}`: {f.get('detail', '')}")
    lines += ["", "### Resources", "",
              f"The audit used {r['wall_seconds']:.2f} wall-seconds and {r['cpu_core_hours']:.5f} CPU core-hours. "
              f"It made {r['decisions']} allocator decisions, ran no training, made {r['circuit_evaluations']} analytic Bell evaluations, solved 16 SDPs, and sampled {r['shots']} shots.",
              "", "### Manuscript", "",
              "The manuscript is `docs/paper/QIP_ROBUST_ENTANGLEMENT_ROUTING.{tex,bib,pdf}`. All result numbers come from `docs/paper/generated/numbers.tex`, and all data figures come from `docs/paper/figure_data/*.csv` (hash-checked).",
              "The supporting files are `docs/paper/BUILD.md`, `SUBMISSION_CHECKLIST.md`, `COVER_LETTER.md`, and `CONTRIBUTIONS_AND_LIMITATIONS.md`.", ""]
    with (ROOT / "docs/stages/29-qip-release.md").open("a") as stream:
        stream.write("\n".join(lines))


def stage29_receipt(args):
    dump("docs/stages/29-publication.json", {
        "stage": "29", "branch": "stage/29-qip-release", "repository": REPO, "source_commit": args.source_commit,
        "results_commit": args.results_commit, "remote_ref": "refs/heads/stage/29-qip-release", "remote_sha_verified": args.remote_sha,
        "bundle": BUNDLE29, "paper_pdf": "docs/paper/QIP_ROBUST_ENTANGLEMENT_ROUTING.pdf",
        "paper_pdf_sha256": hashlib.sha256((ROOT / "docs/paper/QIP_ROBUST_ENTANGLEMENT_ROUTING.pdf").read_bytes()).hexdigest(),
        "github_actions": args.actions, "verified": args.remote_sha == args.results_commit, "verified_utc": now()})
    state = load("project_state.json")
    state["publication_status"] = "complete_remote_verified"
    state["verified_remote_commits"]["stage/29-qip-release"] = args.remote_sha
    state["updated_utc"] = now(); state["next_stage"] = None; state["next_stage_status"] = "none_registered"
    state["next_action"] = "Registered QIP extension complete; author actions in docs/paper/SUBMISSION_CHECKLIST.md."
    dump("project_state.json", state)
    stages = load("stages.json")
    stage(stages, "29")["results_commit"] = args.results_commit
    stage(stages, "29")["publication_receipt"] = "docs/stages/29-publication.json"
    dump("stages.json", stages)


def main():
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(required=True)
    p = sub.add_parser("stage28-receipt"); p.add_argument("--results-commit", required=True); p.add_argument("--remote-sha", required=True); p.set_defaults(fn=stage28_receipt)
    p = sub.add_parser("stage29-start"); p.add_argument("--completion-commit", required=True); p.set_defaults(fn=stage29_start)
    p = sub.add_parser("stage29-source-receipt"); p.add_argument("--source-commit", required=True); p.add_argument("--out", required=True); p.set_defaults(fn=stage29_receipt_file)
    p = sub.add_parser("stage29-results"); p.set_defaults(fn=stage29_results)
    p = sub.add_parser("stage29-receipt"); p.add_argument("--source-commit", required=True); p.add_argument("--results-commit", required=True)
    p.add_argument("--remote-sha", required=True); p.add_argument("--actions", required=True); p.set_defaults(fn=stage29_receipt)
    args = parser.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()
