#!/usr/bin/env python3
"""Validate Stage 00 structure without installing the future research stack.

This checks planning artifacts and links. It does not validate scientific
correctness, authenticate GitHub, or imply that future CLIs exist.
"""

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md", "AGENTS.md", "THIRD_PARTY_NOTICES.md", "stages.json",
    "project_state.json", "references.json", "configs/compute_profiles.json",
    "schemas/run_manifest.schema.json", "docs/EXECUTION_PLAN.md",
    "docs/EXPERIMENT_PROTOCOL.md", "docs/GITHUB_WORKFLOW.md",
    "docs/RESEARCH_ROADMAP.md", "docs/REFERENCE_AUDIT.md",
    "docs/adr/0001-pytorch-stack.md", "docs/stages/00-plan.md",
    "experiments/README.md", ".github/pull_request_template.md",
]
EXPECTED_BRANCHES = [
    "stage/00-plan", "stage/01-reference-port", "stage/02-exploration-controls",
    "stage/03-slam-beliefs", "stage/04-classical-controller",
    "stage/05-critic-ablation", "stage/06-interaction-model",
    "stage/07-shot-noise", "stage/08-confirmatory", "stage/09-release",
]


def main():
    errors = []
    for path in REQUIRED:
        if not (ROOT / path).is_file():
            errors.append(f"Missing required file: {path}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    parsed = {}
    for path in ROOT.rglob("*.json"):
        if ".git" in path.parts or ".venv" in path.parts:
            continue
        try:
            parsed[str(path.relative_to(ROOT))] = json.loads(path.read_text())
        except (ValueError, UnicodeError) as exc:
            errors.append(f"Invalid JSON {path.relative_to(ROOT)}: {exc}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    registry = parsed["stages.json"]
    stages = registry["stages"]
    if [s["branch"] for s in stages] != EXPECTED_BRANCHES:
        errors.append("Stage branch registry differs from the agreed ten-stage plan.")
    seen = set()
    for stage in stages:
        if stage["id"] in seen:
            errors.append(f"Duplicate stage ID: {stage['id']}")
        if not set(stage["depends_on"]).issubset(seen):
            errors.append(f"Missing or forward dependency for stage {stage['id']}")
        seen.add(stage["id"])
    if registry["quantum_advantage_required_to_complete"] is not False:
        errors.append("Completion must permit a correctly established null result.")
    state = parsed["project_state.json"]
    if state["active_stage"] not in seen:
        errors.append("Project state references an unknown stage.")
    if state["repository"]["remote_verified"]:
        if not state["repository"]["created"] or not state["repository"]["url"]:
            errors.append("Remote verification requires a created repository and URL.")
        if not state["verified_remote_commits"]:
            errors.append("Remote verification requires recorded commit evidence.")
    for reference in parsed["references.json"]["repositories"]:
        if not re.fullmatch(r"[0-9a-f]{40}", reference["commit"]):
            errors.append(f"Reference is not pinned: {reference['role']}")
    cap = parsed["configs/compute_profiles.json"]["policy"]
    if cap["restart_resets_budget"]:
        errors.append("Restarting must not reset the consumed compute budget.")

    # Check actual relative Markdown file links. Future paths described in prose
    # or code fences are contracts rather than links to existing deliverables.
    link_count = 0
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        prose = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for destination in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", prose):
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", destination) or destination.startswith("#"):
                continue
            destination = destination.split("#", 1)[0]
            if destination and not (path.parent / destination).exists():
                errors.append(f"Broken link in {path.relative_to(ROOT)}: {destination}")
            link_count += 1

    if errors:
        print("Planning validation FAILED", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Planning validation PASSED")
    print(f"Required files: {len(REQUIRED)}; JSON files parsed: {len(parsed)}")
    print(f"Stages: {len(stages)}; pinned references: 4; relative links checked: {link_count}")
    print("Validated structure, dependency order, provenance fields, and links.")
    print("This is not a training result or proof of GitHub publication.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
