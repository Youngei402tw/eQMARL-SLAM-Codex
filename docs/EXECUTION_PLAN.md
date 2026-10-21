# Codex execution plan

## Status and scope

**Stages 00–02 passed with remotely verified evidence. Stage 03 is running. No policy training or experimental improvement is claimed.**

This document defines how a later authorized Codex implementation session should proceed. A plan is not a background job: work resumes when Codex is actively executing this project. Read this document together with [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md), the [stack decision](adr/0001-pytorch-stack.md), the [reference audit](REFERENCE_AUDIT.md), and the repository's [AGENTS.md](../AGENTS.md).

The core research question is:

> When, if ever, does an entangled interaction critic improve cooperative information gathering under pose uncertainty, compared with equally well developed classical and non-entangled quantum controls?

The intended system uses a conventional 2D SLAM estimator, classical decentralized actors, and a critic that may contain quantum components during training. It does not assume that an entangled training critic enables robots to transmit new map information during execution.

Stage completion means correct implementation and archived evidence. It never requires a quantum win. A valid negative result is a completed research outcome.

## Reference material and interpretation

Use these immutable code snapshots as references, not as code to import unquestioningly:

| Reference | Pinned source | Role |
| --- | --- | --- |
| Original eQMARL implementation | [news-vt/eqmarl, 6b1b2e3](https://github.com/news-vt/eqmarl/tree/6b1b2e3817f661da81aad5e7188a362380dbdd1d) | Circuit architecture, paper configuration, reference behavior |
| Older mapping implementation | [eQMARL-SLAM, 60388c4](https://github.com/Youngei402tw/eQMARL-SLAM/tree/60388c4641a6ce23b4b750f6bde4bc97b7621036) | Known-pose exploration and engineering lessons |
| Preserved NEW baseline | [eQMARL-SLAM-NEW, d16916f](https://github.com/Youngei402tw/eQMARL-SLAM-NEW/tree/d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f) | Historical faithful-pilot protocol |
| Active-SLAM improvement branch | [eQMARL-SLAM-NEW, 4ff242a](https://github.com/Youngei402tw/eQMARL-SLAM-NEW/tree/4ff242a69f33ac76a3127b6a960552f6b9a629d5) | Documented experiments, remaining modeling and evaluation limitations |

The [paper](https://arxiv.org/html/2405.17486v2) concerns a split quantum critic with initial inter-agent entanglement, local encoding, and joint measurement. Its MiniGrid experiment uses independent environments. The new project must distinguish architecture parity, training corrections, and a new cooperative SLAM task.

Preserve attribution and applicable licenses for any reused source. Record substantive departures from a reference configuration. Do not claim independently reproduced historical results when only a committed summary was available.

## Non-negotiable scientific contracts

1. **Information boundary:** an actor receives its own belief/history and explicitly delivered messages. A crop of a globally fused map is a shared-information condition and must be labeled and charged accordingly.
2. **Real uncertainty:** the SLAM stage must estimate uncertain poses and unknown map variables. Exact positions supplied to an actor define a separate known-pose exploration control.
3. **Useful revisits:** an informative revisit may improve localization without discovering new cells. Reusing the same measurement must not create fictitious independent evidence.
4. **Common scaffolding:** compared learning methods share the estimator, candidate actions, planner, actor architecture, task definition, and evaluation conditions unless a specific ablation says otherwise.
5. **Matched entanglement control:** qfCTDE is not the no-entanglement version of eQMARL. Include the identical split circuit with only its input entanglement disabled.
6. **Honest resources:** count environment steps, optimization updates, classical communication, quantum shots/circuit calls, and wall time separately. A small central readout does not imply a small total system.
7. **Frozen final evaluation:** development decisions use training/development/validation data. Test policies are frozen, and the final test set cannot become another tuning set.
8. **Complete reporting:** retain failures, interrupted runs, and negative findings. Do not search indefinitely for a quantum advantage.

## Framework direction

The intended implementation uses PyTorch for actors, critics, recurrent state, optimization, and experiment tooling. Use the PyTorch-compatible quantum backend selected by the [stack decision](adr/0001-pytorch-stack.md) and verified in Stage 01. Preserve a backend-independent critic interface so exact statevector, finite-shot, and independent reference calculations can be compared.

Keep TensorFlow/Cirq dependencies isolated if they are needed to extract reference fixtures. The main project must not require the legacy TensorFlow Quantum stack merely to run a classical baseline. Do not perform a line-by-line framework translation of upstream training quirks.

Use GTSAM for the initial factor-graph SLAM implementation if the supported runtime passes a small compatibility probe. The [official planar example](https://gtbook.github.io/gtsam-examples/PlanarSLAMExample.html) is a suitable starting point. An EKF can be an explicitly versioned alternative if GTSAM is unavailable; it cannot silently replace the estimator in only some compared methods.

## Branches and dependencies

Only the Stage 00 branch is required now. Create future branches when their stages begin. The stage index may list a planned branch before it exists, but must not imply that experiments have run.

| Stage | Exact branch | Dependency and outcome |
| --- | --- | --- |
| 00 | stage/00-plan | Planning baseline |
| 01 | stage/01-reference-port | 00; tested circuit and interface contracts |
| 02 | stage/02-exploration-controls | 01; trustworthy exploration controls |
| 03 | stage/03-slam-beliefs | 02; verified 2D estimator and sensing |
| 04 | stage/04-classical-controller | 03; common controller and strong classical baselines |
| 05 | stage/05-critic-ablation | 04; matched critic comparison |
| 06 | stage/06-interaction-model | 05; structured interaction study |
| 07 | stage/07-shot-noise | 05 and the frozen candidate from 06, when used |
| 08 | stage/08-confirmatory | 05 and all development stages relevant to its registered claims |
| 09 | stage/09-release | 08 for a complete core research release; otherwise a clearly labeled partial release |

Stage 06 can pass forward with a valid null result and the simpler candidate retained. Stage 07 may be blocked by backend or compute availability without preventing a correctly labeled ideal-simulation comparison in Stage 08. A robustness claim requires completion of the relevant Stage 07 experiments. Optional integration work is not a prerequisite for a core 2D result.

Each stage records its parent implementation commit. Branch from that verified commit, not blindly from the current main branch. Retain completed stage branches. Use a PR to show the stage's changes, outcomes, limitations, and gate status. Follow AGENTS.md for integration into main; updating a project index must not overwrite an unfinished stage.

## Standard execution loop

Before working on any stage:

1. Inspect the remote branch and local tree; preserve unrelated changes.
2. Read the latest stage record and determine whether work is planned, running, completed, or blocked.
3. Resolve the source commit, input artifacts, protocol version, dataset manifests, and dependency lock.
4. Check whether the intended run already exists. Resume or report it rather than silently duplicating it.
5. Run the stage preflight and a throughput measurement.
6. Select a feasible budget within the declared caps before comparing outcomes.
7. Commit the implementation and resolved configuration before the scientific run.
8. Execute the cheapest check that could falsify the implementation.
9. Run the authorized bounded experiments, checkpointing and recording failures.
10. Publish metrics, analysis, checkpoints or verified artifact references, and the gate report.
11. Verify the remote files/commit and update the stage record.

The intended future command interface is shown below. These commands are **contracts to implement**, not commands asserted to work in the planning-only repository:

~~~bash
python -m eqmarl_slam.cli preflight --stage 01 --profile smoke
python -m eqmarl_slam.cli run --config <committed-config-path>
python -m eqmarl_slam.cli resume --run <run-id>
python -m eqmarl_slam.cli analyze --stage 05 --protocol <protocol-id>
python -m eqmarl_slam.cli verify --run <run-id>
~~~

A CLI must return a nonzero exit status on failed correctness checks, invalid configuration, missing artifacts, or inconsistent resumes. Resource exhaustion must be recorded distinctly from scientific failure.

The [experiment protocol](EXPERIMENT_PROTOCOL.md) controls budget semantics; [compute_profiles.json](../configs/compute_profiles.json) is the machine-readable representation. Resolve any disagreement before execution. The initial stage cap is 24 cumulative active execution wall-hours across runs, attempts, and resumed sessions. A restart, new branch, or renamed run must not reset consumed budget. More restrictive per-run and host limits also apply. Record CPU core-hours and GPU device-hours separately; they are not interchangeable budget units.

## Stage 00 — Plan and repository foundation

### Preflight

- Confirm the intended repository owner/name and avoid modifying the prior projects.
- Inventory available tools, local runtime, and existing authorized compute. Do not assume GPU or cluster access.
- Record the source snapshots and distinguish reference-derived facts from proposed experiments.

### Deliverables

- This plan and the experiment protocol.
- Project instructions, architecture decision, stage index, and dependency/resource policy.
- A clear status stating that model implementations and training results are not yet available.
- A branch and initial commit that the user can inspect.

### Gate

Every later stage has a branch name, scope, acceptance criteria, failure handling, and evidence requirements. No planned run is described as executed. Relative documentation links resolve once the planning scaffold is complete.

### Failure and resume

If remote publication is blocked, finish the reviewable local planning package, record the exact missing capability, and do not claim repository creation or a successful push. Resume by verifying the remote state before repeating a mutation.

## Stage 01 — Reference circuit port and model contracts

### Preflight

- Select and lock compatible framework/backend versions.
- Establish a classical-only installation path.
- Extract small fixed reference cases from pinned source without importing its optimizer behavior.
- Declare dtype, qubit indexing, basis ordering, rotation order, and encoding transformations.

### Implementation and experiments

- Implement critic interfaces, state/observation adapters, exact circuit evaluation, and serialization.
- Check Bell preparation, local variational blocks, global parity, and circuit batching.
- Compare independent 2-, 4-, and 8-qubit fixtures where practical.
- Compare input and parameter gradients with finite differences at nondegenerate points.
- Test repeated identical inputs and heterogeneous batches.
- Keep a named reference readout for parity checks and a separate affine head for corrected experiments.
- Verify the maximally mixed local-marginal property for fully paired Bell states followed only by local unitaries. Do not replace useful cross-agent readouts with local-only expectations.
- Measure throughput and memory for candidate circuit sizes.

### Gate

Small float64 fixture expectations must agree within atol=1e-8. State norm error must be at most 1e-10 for the independent float64 reference. Gradient checks use combined absolute/relative tolerance, initially atol=1e-6 and rtol=1e-3, together with finite-difference step-size convergence. See the [stack decision](adr/0001-pytorch-stack.md) for the fixture contract. A cross-framework comparison involving a float32 backend may declare a separately justified tolerance before checking results; it does not replace the float64 unit-fixture gate.

Tolerances may change only for a documented numerical reason. A legitimately zero derivative at a symmetry point is not a failure; disconnected encoders or unexpected constant cross-agent outputs are.

### Failure and resume

Investigate ordering, normalization, parameter reuse, and dtype before changing architecture. Keep at most two diagnosis/patch cycles for the same unexplained discrepancy. If a backend remains incompatible, record the blocker and evaluate an explicitly documented alternative. Do not proceed to quantum performance claims until parity and gradient checks pass.

## Stage 02 — Controlled exploration and evaluation

### Preflight

- Record the known-pose versus uncertain-pose condition.
- Make observation sharing, sensor visibility, map bounds, collision rules, and horizon explicit.
- Require every evaluator to load the same resolved environment configuration.

### Implementation and experiments

- Establish a cheap grid exploration control using lessons from the prior implementation.
- Implement own-map observations and separately labeled shared-map observations.
- Add an information-isolation check: changing a teammate's private map must not change an actor input when no message was delivered.
- Compute physical visibility coverage independently from estimated-map coverage.
- Repair and validate frontier selection, unreachable targets, and the current-cell target case.
- Run scripted trajectories, random actions, a reactive policy, and frontier assignment.
- Verify that sampled versus greedy evaluation modes are explicit.
- Define whether the episode horizon is intrinsic. If intrinsic, expose remaining budget; if only a collection cutoff, preserve appropriate bootstrapping.

### Gate

No information-boundary or metric inconsistency remains. Heuristics make progress on simple solvable maps. A learned-policy score is not compared to a heuristic score from another reward, target, or time-limit protocol.

### Failure and resume

If the task is unsolvable, a heuristic stalls, or physical and estimated coverage disagree unexpectedly, inspect the environment before training longer. Create a new protocol revision for semantic fixes. Keep old outcomes labeled with their original protocol.

## Stage 03 — Genuine 2D SLAM beliefs

### Preflight

- Validate a minimal estimator installation.
- Specify units, gauge fixing, initial-pose priors, data association, process noise, and observation models.
- Start with known landmark identities and unknown landmark coordinates.
- Specify message contents, delivery schedules, measurement IDs, and local versus shared factor ownership.

### Implementation and experiments

- Implement noisy body-frame odometry and range/bearing observations, not true world-frame displacement passed directly to the policy.
- Maintain uncertain poses and map variables.
- Validate one-robot trajectories before two-robot cooperation.
- Test a straight path, a closed loop, revisiting a known landmark, and a relative observation between robots.
- Verify that closure factors can revise historical estimates.
- Keep actor inputs separate from ground-truth state used for simulation/evaluation.
- Deduplicate repeated factors and handle shared information consistently.
- Compare current-pose error, trajectory error, map error, and the estimator's uncertainty.
- Permit genuine uncertainty improvement in known space. Do not require new map cells as a universal condition for useful sensing.

### Gate

A controlled observable revisit reduces a relevant marginal relative to the same-step prediction/no-measurement control, or improves another declared estimation measure. Process noise may make the posterior larger than an earlier-time covariance; that alone is not a failed measurement update. An uninformative observation or duplicate measurement ID cannot invent new independent information. Gauge/coordinate handling is consistent. Errors are assessed over a fixed reference convention rather than hiding drift through per-step alignment.

### Failure and resume

Diagnose estimator behavior on scripted trajectories before connecting RL. If GTSAM is blocked, a versioned EKF alternative needs separate protocol/results. A standard current-pose-and-landmark EKF does not retain historical pose states and cannot pass the historical-trajectory correction gate. That gate requires a smoother or an explicitly augmented historical-state estimator. A standard EKF may support a narrower current-pose/map uncertainty study with separately defined trajectory metrics and no claim of historical smoothing. Do not substitute exact pose. If joint-map exchange cannot preserve consistent information, keep local beliefs and restrict the claimed cooperative sensing scope.

## Stage 04 — Common high-level recurrent controller

### Preflight

- Freeze the estimator and task interface for the controller pilot.
- Choose a small candidate-action set generated from permitted information.
- Define infeasible-action masks, maximum goal duration, interrupted goals, and memory resets.

### Implementation and experiments

- Build a shared classical encoder/GRU actor with uncertainty and budget summaries.
- Choose among frontier, revisit, rendezvous, and continuation actions.
- Execute actions using the same planner/controller for every method.
- Aggregate rewards and discount by actual action duration.
- Implement random/reactive, frontier-assignment, and uncertainty-aware heuristics.
- Implement a classical recurrent MAPPO baseline with detached targets and explicit terminal/truncation semantics.
- Run simple learning tests and bounded pilots; report primitive steps, not only episode counts.

### Gate

Correctness tests pass; the task is demonstrably tractable for a competent heuristic. Classical learning is stable enough to make a meaningful comparison on the simple conditions. There is no requirement for the learned policy to beat every heuristic.

### Failure and resume

If all learned policies collapse, check observation sufficiency, action masks, returns, entropy, reward scale, update frequency, and normalization before adding quantum models. Tune within the common declared budget. If classical learning remains unstable, record that limitation and restrict subsequent conclusions; do not call quantum superiority from a broken baseline.

## Stage 05 — Matched critic ablations

### Preflight

- Freeze the shared actor architecture, planner/estimator interfaces, and pilot task protocol. Actor weights remain trainable independently within each method; only a separately registered fixed-encoder/actor diagnostic freezes learned weights.
- Register the method matrix and tuning budget.
- Check that learned local compression remains trainable when intended; a frozen quantum reupload scale is not a frozen classical encoder.

### Implementation and experiments

- Compare compressed classical split, centralized classical, and classical bilinear critics.
- Compare split quantum circuits with and without input entanglement.
- Retain qfCTDE as a separate topology comparison.
- Use affine value heads and common return normalization in the corrected comparison.
- First fit a fixed training/validation dataset of return targets.
- Diagnose target-sign coverage, output variance, gradient norms, encoder connectivity, and learning-rate behavior.
- Then run bounded RL pilots with a common actor/training stack.
- Preserve any architecture-faithful/A2C reference experiment under a separate protocol; do not mix it into the corrected MAPPO leaderboard.

### Gate

The comparison changes what it claims to change. Every method has meaningful gradients where expected and can pass small-dataset fitting checks appropriate to its capacity. All runs and tuning outcomes are archived. A valid null result passes the stage.

### Failure and resume

Fix observed numerical/implementation failures within the retry policy. If a correctly implemented critic underfits, report that result rather than enlarging only that method. Any expanded capacity/tuning study must be registered as a new symmetric comparison.

## Stage 06 — Structured interaction model and attribution

### Preflight

- State the mechanism being tested before selecting more complex circuits.
- Separate additive local value from pairwise cooperative interactions.
- Register conditions with/without uncertainty, complementary observations, rendezvous utility, and limited communication.

### Implementation and experiments

- Test local value terms plus cross-agent quantum interaction features.
- Use independent small pair circuits for selected edges; do not assume one qubit can be maximally Bell-entangled with many neighbors simultaneously.
- Compare identical local terms and heads with classical bilinear or graph interactions.
- Include no-interaction and no-input-entanglement controls.
- Examine 2 robots first; add 3 or 4 only after throughput and symmetry tests.
- Record classical messages required by local-value terms and edge selection.
- Select a final candidate using validation performance and resource cost together.

### Gate

The experiments distinguish quantum components from useful classical interaction structure. Interpret gains and losses conditionally on task structure. A result that a classical bilinear critic matches eQMARL is scientifically valid.

### Failure and resume

If the model increases cost without a repeatable validation benefit, retain the simpler candidate and document the negative result. Do not conduct unbounded Bell-state, depth, or topology searches. Changes to the selected final candidate require a new recorded validation decision.

## Stage 07 — Finite shots and declared noise

### Preflight

- Confirm that the backend's gradient method is valid for finite-shot/noisy evaluation.
- Register a small method subset and noise model.
- Distinguish model-level noise experiments from a full hardware/network claim.

### Implementation and experiments

- Compare exact expectations with initial shot settings such as 64, 256, and 1024 on fixed diagnostic data.
- Check sampling error, gradients, value error, and compute cost.
- Use a small declared noise grid, initially one noise family at a time.
- Run a limited matched training pilot if the budget permits.
- Count actual circuit calls, shots, retries, and any discarded/lost results.
- Report training resources separately from the classical deployed actor.

### Gate

Sampling behavior and resource accounting are credible. The study reports where performance becomes unreliable or too expensive. A frozen-critic sensitivity result is labeled as such and is not misrepresented as a trained finite-shot policy result.

### Failure and resume

Stop an unstable or expensive regime at its predefined cap, preserve the checkpoint, and report the failure threshold. If unavailable backend features prevent the stage, mark it blocked and narrow later claims to the completed ideal-simulation scope.

## Stage 08 — Locked confirmatory comparison

### Preflight

- Commit the complete analysis plan, selected methods, configurations, training budget, seeds, checkpoint-selection rule, test manifest, and metric definitions.
- Select at most four learned methods for the confirmatory study: a strong classical method, matched split quantum no-entanglement, and eQMARL are required; a fourth method is optional and chosen from validation evidence.
- Verify that final test conditions were not used to choose models or hyperparameters.
- Estimate the whole study's cost and choose a feasible fixed budget before observing test outcomes.

### Implementation and experiments

- Train independent seeds under the locked protocol.
- Evaluate frozen policies on paired held-out maps and disturbances.
- Use the declared sampled/greedy action mode.
- Analyze seed-level results, paired differences, uncertainty intervals, failures, and resource tradeoffs.
- Report physical coverage, mapping accuracy, trajectory error, communication, collisions, and coverage–accuracy success criteria.
- Keep best-checkpoint and final-checkpoint evaluations distinct if both are registered.

### Gate

All planned runs have a recorded outcome. Every table traces to raw metrics. No test-time updates or outcome-driven retuning occurred. Failure to detect a difference is not called equivalence; any equivalence claim would require a separately prespecified margin and analysis.

### Failure and resume

Resume interrupted training from compatible checkpoints. Correct a mechanical analysis error transparently and regenerate all affected outputs. A substantive post-test model change starts a new protocol and uses a new reserved test set; retain the first result.

## Stage 09 — Reproducible release and handoff

### Preflight

- Audit source/config/data/result/checkpoint identities.
- Verify remote artifact availability and the absence of unsupported claims.
- Identify what was completed, omitted, blocked, or inconclusive.

### Implementation and experiments

- Recreate the runtime in a clean environment.
- Reproduce a compact smoke experiment.
- Regenerate final analysis from archived raw data without retraining.
- Verify artifact hashes and frozen-checkpoint loading.
- Publish final reports, limitations, resource tables, and a concise continuation guide.
- Record the exact completion commit for every stage.

### Gate

A fresh session can identify the current state, rerun the smoke path, and regenerate reported analysis. Required evidence is stored on GitHub with verified identities. A partial release is explicitly labeled partial.

### Failure and resume

Repair missing metadata or links before calling the archive complete. If large-artifact publication is blocked, keep the available evidence and report blocked_publication. Do not delete old stage branches or negative experiments to improve presentation.

## Scope control and finish conditions

The core project ends when the 2D task, fair critic comparison, and confirmatory evaluation are complete and archived, even if eQMARL does not win.

Defer ROS/Gazebo integration, ambiguous association, extensive scaling, 3D perception, and execution-time entangled actors to separately planned branches. They need their own hypotheses and resource budgets. A useful first paper can concern when entanglement helps or fails; photorealistic rendering is not a substitute for causal evidence.

At the end of every active Codex session, record the last verified commit, next unambiguous action, active run IDs/processes when applicable, blockers, and remaining budgets. Never imply that work continues unattended after the session unless an actual running job or authorized schedule exists.
