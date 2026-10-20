# Stage 00 — plan and repository scaffold

Status: **completed and remotely verified.** The user-created repository is public. The planning validator output is in [00-validation.txt](00-validation.txt), and [00-publication.json](00-publication.json) records the verified planning commit and all-file hash check. The subsequent index on `main` records the immutable final Stage 00 completion commit, avoiding a self-referential SHA.

## Question

Can the proposed project be executed and reviewed stage by stage, with a clear reference, fair comparisons, meaningful SLAM uncertainty, bounded experiments, and a permanent GitHub record?

## Deliverables

The scaffold provides a ten-stage execution plan; Codex instructions; a PyTorch/PennyLane architecture decision; a reference audit; experiment, statistics, and archive protocols; a branch registry; compute profiles; a run-manifest schema; a research ranking; and a standard-library planning validator.

Reference commits are frozen in `references.json`. Future dependency versions will be resolved and locked only after a real compatibility smoke test in Stage 01. No upstream source code or checkpoints have been copied into this scaffold.

## Decisions

- Start with 2D and use grid exploration only as a diagnostic before genuine estimator-based SLAM.
- Preserve a faithful reference port and clearly version subsequent corrections.
- Use the same classical recurrent actor, action interface, estimator, evaluation protocol, and encoder budget for critic comparisons.
- Require strong classical, matched no-entanglement, and entangled controls. Add pairwise interaction models only as a separately identified research hypothesis.
- Record every experiment outcome, including failures and null results, with source commits and raw evidence.
- Create stage branches on stage start and retain them permanently; archive code and results under immutable commits.
- Use provisional bounded compute profiles. Estimate actual throughput before allocating a run; no paid hardware provisioning is implied.

## What has and has not been executed

This stage prepares research instructions and files. It has not trained a policy, ported the quantum implementation, installed the research dependency stack, validated a SLAM estimator, or measured a quantum advantage. The validator output is planning evidence only.

The GitHub plugin verified the repository and both initial branches. The published planning commit contains 20 files; all remote Git blob hashes matched the prepared local bytes. The completion receipt adds the publication record. The final stage branch is retained after integration into `main`.

## Exit gates

1. Required files exist and the planning validator passes.
2. The plan accounts for reference parity, environment/estimator correctness, fair baselines, resource limits, statistical inference, artifact publication, and failure recovery.
3. The target repository exists, Stage 00 content is published, and remote files/commits have been verified.
4. The next stage has a precise preflight: resolve the Python 3.11 target stack, pin the environment, implement numerical fixtures, and record reference/corrected behavior separately.

A publication blocker leaves gate 3 incomplete even if the design is ready. Resolve that blocker before reporting Stage 00 as remotely complete.

## Next action

On an instruction to execute Stage 01, create `stage/01-reference-port` from the completion commit recorded in `main:project_state.json`. Read the architecture decision and Stage 01 preflight, resolve the compatible Python/PyTorch/PennyLane environment, and implement the numerical reference fixtures and model contracts before any learning experiment.
