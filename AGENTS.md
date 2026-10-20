# Codex execution instructions

## Goal and authority

Investigate eQMARL for cooperative active SLAM, starting in 2D, with fair classical controls and a permanent GitHub record of each stage and experiment. Use the original eQMARL repository as the reference, with PyTorch and PennyLane as the new implementation stack.

The initialization request covers Stage 00. Do not begin the entire training program merely because this file exists. Once the user asks to execute the plan or a stage, continue through the authorized scope without asking for permission at every routine, reversible step. Existing user instructions take precedence over this file. Provisioning paid compute, publishing credentials, destructive history changes, or expanding access are outside routine research execution.

## Read before every work session

1. Read this file, `project_state.json`, `stages.json`, the active stage in `docs/EXECUTION_PLAN.md`, and its existing report under `docs/stages/`.
2. Read `docs/EXPERIMENT_PROTOCOL.md`, `docs/GITHUB_WORKFLOW.md`, and relevant architecture decisions. Inspect reference code at `references.json` commits when the task depends on it.
3. Inspect the working tree and remote state. Preserve user changes. Reuse an existing stage branch; otherwise create it from the exact recorded dependency commit. Never force-push or delete completed stage branches.
4. Verify what actually exists. Documentation about a future CLI, dependency lock, experiment, or remote branch does not mean it is implemented or published.

## Stage preflight, before an expensive or consequential action

- State the stage question, the smallest experiment that could falsify the proposed explanation, the concrete deliverables, and the exit/failure gates.
- Record source commit, dependency completion commit, protocol identifier, complete resolved config, dataset/split hashes, seed streams, and compute profile. Commit code/config before training; do not train from an unrecorded dirty tree for reportable runs.
- Check estimator, actor, critic, reward, and evaluator information boundaries. Truth is available only to the simulator and explicitly declared training/evaluation components. Actors cannot silently receive global maps or teammate state.
- Resolve actual dependency compatibility in a clean environment before pinning it. Use the separate upstream reference environment for legacy TensorFlow. Never edit a wheel filename or suppress dependency constraints to claim compatibility.
- Estimate memory, runtime, circuit evaluations, and shots from a short benchmark. Use the cumulative cap in `configs/compute_profiles.json` and the experiment protocol; a restart does not reset the budget.
- Confirm the output path and GitHub publication mechanism. Small raw metrics/configs must be versioned; large artifacts require durable storage, hashes, and verified retrieval. Archive availability is a stage gate.
- Implement and run the cheapest meaningful correctness checks first. Broaden testing only to resolve a concrete risk or required gate.

## Implementation rules

- Keep a separately named faithful reference path and corrected path. Never label an architectural or optimizer change a faithful reproduction.
- All critic comparisons use the same actor/controller, observation contract, encoder dimensions, rollout semantics, and declared evaluation suite. Tune each method within an equal search budget; identical learning rates are not required.
- A no-entanglement ablation must differ only in the declared entanglement intervention. A different quantum architecture such as qfCTDE is an additional comparator, not that ablation.
- Keep analytic simulation, finite-shot simulation, noisy simulation, and physical hardware claims separate. Count backward-pass quantum evaluations and shots.
- Use joint observables that can retain cross-agent information. For a maximally entangled two-party state with local unitaries only, single-party traceless marginals are zero; test this invariant rather than assuming local readouts contain encoded observations.
- A real SLAM stage needs an estimator with pose/landmark uncertainty and informative revisits. A clipped diagonal uncertainty heuristic or known true pose is an exploration diagnostic and must be labeled accordingly.
- Define primitive-step and macro-action duration semantics, terminal vs time-limit bootstrapping, RNN resets, and masking explicitly. Version changes to reward, information, action space, estimator backend, success threshold, horizon, or metric denominators.
- Do not change the final test set, drop failed seeds, or extend a budget after seeing a disappointing final result. New hypotheses require a new development protocol.

## Experiment execution and publication

1. Pre-register the run in the stage run index with a unique ID and status `planned`.
2. Commit and publish the source/config commit A; verify the remote SHA. Record A in the run manifest.
3. Run the experiment within its cap, checkpointing and writing append-only machine-readable metrics. Use separate training, development, validation, and final-test manifests.
4. Archive every outcome, including `failed`, `stopped_budget`, and null findings. Write the manifest, resolved config, raw metrics, diagnostics, summary, artifact inventory, and short report. Never create fabricated metric rows for a run that did not happen.
5. Validate the bundle and verify checkpoint hashes. Commit the archive as results commit B on the stage branch; record A in the bundle and B in the external stage index or a later receipt, avoiding a self-referential commit hash.
6. Push B through the connected GitHub plugin or the already authorized Git transport. Read back the branch head and representative files, and verify large artifact retrieval. Mark publication complete only after evidence exists.
7. Update the stage report and `project_state.json`. Keep source and results from parallel experiments on `exp/NN-...` branches until integrated into the owning stage branch.

Do not blanket-ignore `experiments/` or all checkpoint files and then report an experiment as archived. GitHub Actions artifacts with an expiry date are not the sole archive.

## Failure and recovery

- On an unexplained repeated failure, diagnose, patch, and retry at most twice for that failure mode, then record the blocker and continue independent useful work.
- A correctness failure blocks dependent scientific comparisons. A null result does not: finish the relevant diagnostic and pass a documented null finding forward.
- A compute limit triggers a checkpoint and `stopped_budget`, with consumed resources and the proposed next experiment recorded. Do not provision paid hardware or multiply the search budget silently.
- A failed push/upload triggers `blocked_publication`; preserve the local bundle and retry the supported route. Do not claim remote completion from a local commit.
- If a required permission or user choice truly blocks work, first finish all authorized preparation and ask for the specific missing action, identifying its cause.

## Stage completion

A stage is complete when its correctness gates pass, its allowed positive/null/negative outcome is explained, and all required artifacts are remotely verified. Record the immutable completion commit and a reviewable stage report. Open a stage PR if supported; do not request unrelated reviewers or send messages. Preserve the branch after integration. Create the next stage only from its recorded dependency state.

At the end of each session, report what changed, what was actually run, the results and their limits, branch and commit links, publication status, consumed budget, and the precise next action. Never imply that future stages or experiments are already running.
