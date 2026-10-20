# GitHub and experiment history

## Repository initialization

The user created the public repository [`Youngei402tw/eQMARL-SLAM-Codex`](https://github.com/Youngei402tw/eQMARL-SLAM-Codex). The connected GitHub plugin verified the repository, its empty initial state, and write access on 2026-10-20. Repository creation and browser sign-in are resolved; publish through the existing GitHub connection.

Initialize `main` with the bootstrap commit, create `stage/00-plan` from it, publish the complete planning tree, and verify the remote contents. Integrate the verified stage through a reviewable PR while retaining the stage branch. Record the immutable stage completion commit in a subsequent index update on `main`, avoiding a self-referential hash. Later stages use the same publication contract. Preserve user changes and existing remote history.

## Branch contract

`main` contains the stable index and integrated completed work. `stages.json` lists the exact future branch names and dependency relationships.

Create each stage branch when the stage begins. It is a durable snapshot of that stage's code, configs, raw experiment records, analysis, and report. Branches for future stages are planned names until creation is verified. Never delete stage branches after merging.

For a parallel experiment that would otherwise mix incompatible implementations, use `exp/<stage>-<short-purpose>`. Merge its evidence and the selected implementation into the owning stage branch. Keep the run's original source commit in its manifest. A later stage can revise code without rewriting an earlier stage's results.

Prefer one reviewable PR per completed stage. The PR records the problem, changes, actual runs, gate outcomes, results including null findings, and remaining limits. Routine integration can follow existing user authorization and passed gates; this file does not create an extra per-stage user approval requirement. Respect actual branch protection if configured.

## Commit provenance without circular hashes

1. Commit A contains the exact code and resolved configuration used to run an experiment. Publish A before a reportable run begins.
2. The run manifest records A. Any correction after the run has started gets a new run ID and source commit.
3. Commit B adds that run's output bundle. The bundle cannot contain its own future commit B hash.
4. A subsequent stage-index update or PR records B, or the GitHub branch/commit URL serves as the results receipt. Neither A nor B is changed to manufacture a self-reference.

For stage completion, use a later index update on `main` or a PR record to point to the immutable completion SHA on the retained stage branch. If annotated tags or releases are supported in the execution environment, they may add convenient labels; branch and commit provenance must work without them.

## Mandatory run archive

Each executed run is stored under `experiments/<stage>/<run_id>/` with:

- `manifest.json`: identity, provenance, protocol, seed streams, runtime, status, and resource use.
- `resolved_config.yaml`: every effective setting, including defaults and environment overrides.
- `metrics.jsonl`: raw episode/update metrics; no only-smoothed replacement.
- `summary.json`: machine-readable endpoints and the aggregation rule.
- `diagnostics.json`: relevant correctness, gradients, estimator, and resource checks.
- `artifact_manifest.json`: each artifact's relative path or durable GitHub URL, size, SHA-256, purpose, and retention requirement.
- `report.md`: hypothesis, outcome, interpretation, limitations, and a command to reproduce using the implemented CLI.

Failed runs still have a manifest, failure reason, diagnostics, resource use, and any valid observations. If no data were produced, record that fact rather than inventing an empty success. Illustrative templates live outside `experiments/` and are never included in scientific aggregates.

## Artifact size and durability

Commit all small raw metrics, configs, seed manifests, analysis code, summaries, and reports to Git. The project policy uses 10 MiB as the default per-file review threshold; this is a project choice, not a claim about GitHub's service limit.

Retain the selected and final checkpoint for every reported training run, plus sufficient optimizer/RNG/estimator state to resume interrupted runs when continuation is claimed. Small checkpoints can be committed. Larger checkpoints, videos, and sensor logs may use Git LFS or release assets only after the actual environment's upload, quota, and read-back route are verified. Store hashes and immutable references in Git. A release or LFS pointer without a retrievable object is incomplete publication.

Do not rely solely on temporary CI artifacts, expiring notebook storage, a private local cache, or a summary image. If durable artifact storage is unavailable, stop before deleting local artifacts, publish all small evidence, and mark the stage `blocked_publication` until the required files are retrievable. Do not incur new storage charges without existing authorization.

## Verification checklist

Verify the branch head matches the intended results commit, fetch the committed manifest and at least one metrics/summary file, and verify required large artifact hashes from a fresh retrieval. Record the check time, commit, and locations in the stage receipt. If the plugin lacks a needed operation, use an already authorized transport or report the concrete capability gap.

Use normal non-force ref updates with an expected old SHA when available. If the remote advanced, fetch and integrate it; do not overwrite other work. Exclude credentials, access tokens, local environment files, and unrelated user data.

## CI scope

Stage 01 adds a small CPU CI workflow for import checks, numerical fixtures, schema checks, and deterministic environment diagnostics. Pin and verify actual official action versions then. Full training is an explicit research run, not an automatic side effect of every push. CI must not silently create paid GPU jobs or store final evidence only in expiring artifacts.
