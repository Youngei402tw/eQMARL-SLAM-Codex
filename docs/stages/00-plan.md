# Stage 00 — plan and repository scaffold

Status: **design prepared and structurally validated; remote publication verification in progress.** The repository creation blocker is resolved. The planning validator output is recorded in `00-validation.txt`, and `project_state.json` records publication evidence.

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

Remote repository creation, branch creation, and publication must be verified independently. A local directory is not a GitHub repository. Consult `project_state.json` for the actual publication status and blocker, if any.

## Exit gates

1. Required files exist and the planning validator passes.
2. The plan accounts for reference parity, environment/estimator correctness, fair baselines, resource limits, statistical inference, artifact publication, and failure recovery.
3. The target repository exists, Stage 00 content is published, and remote files/commits have been verified.
4. The next stage has a precise preflight: resolve the Python 3.11 target stack, pin the environment, implement numerical fixtures, and record reference/corrected behavior separately.

A publication blocker leaves gate 3 incomplete even if the design is ready. Resolve that blocker before reporting Stage 00 as remotely complete.

## Next action

Finish publication if needed. On an instruction to execute Stage 01, start `stage/01-reference-port` from the verified Stage 00 completion state, read the architecture decision, and run the compatibility and numerical preflight before any learning experiment.
