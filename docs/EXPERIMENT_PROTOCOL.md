# Experiment and publication protocol

## Status

This is a prospective protocol for the [execution plan](EXECUTION_PLAN.md). **The current repository setup is Stage 00 planning. No experiments described here have been run by virtue of this document.**

All numeric budgets and tolerances below are proposed defaults. They are limits and starting points, not claims of adequate statistical power, achievable runtime, or optimal hyperparameters. This document controls the scientific and resource semantics; [compute_profiles.json](../configs/compute_profiles.json) is their machine-readable representation. Resolve discrepancies before running an experiment.

## 1. Protocol identity and immutable experiments

Every scientific experiment must have:

- A stage ID.
- A protocol ID and revision.
- A method ID.
- A stable run ID.
- A committed resolved configuration.
- An implementation/source commit.
- Dataset/scenario manifests.
- Independent seed streams.
- A status, budget, and artifact manifest.

Run IDs and execution manifests must follow the authoritative [run-manifest schema](../schemas/run_manifest.schema.json), currently schema_version `0.1.0`. Suggested run ID:

~~~text
05-pilot_belief-v1_eqmarl_seed-0002_cfg-a1b2c3d4
~~~

The configuration hash is part of the identity. A rerun with the same scientific specification uses an explicit attempt number rather than replacing the first outcome. An interruption resumed from its checkpoint retains the run ID and records additional execution segments.

Change the protocol revision whenever changing rewards, success criteria, observation sharing, estimator semantics, sensor noise, horizon, action granularity, actor architecture, training algorithm, metric definition, checkpoint selection, or test scenario allocation. Cosmetic reporting changes do not require a new scientific protocol, but retain analysis provenance.

Do not combine metrics from different protocols into one ranking. The older 90%-coverage pilot and the later 99%-coverage task are separate historical examples, not interchangeable observations.

## 2. Reference parity versus corrected experiments

Use separate protocol families:

| Family | Purpose | Allowed interpretation |
| --- | --- | --- |
| reference-circuit-v1 | Verify circuit math and output conventions against pinned eQMARL source | Architecture parity |
| reference-training-v1, only if implemented | Reproduce a precisely documented historical training configuration | Scoped reproduction, including disclosed source quirks |
| exploration-controls-v1 | Validate information boundaries and exploration metrics | Known-pose/grid diagnostics |
| belief-slam-v1 | Genuine noisy-pose 2D SLAM with explicit communication | Active-SLAM experiments |
| critic-ablation-v1 | Matched quantum/classical critics with common scaffolding | Component attribution |
| interaction-study-v1 | Local value plus pairwise interactions | Mechanism and scaling analysis |
| shot-noise-v1 | Declared finite-shot/noise resource conditions | Robustness within the tested model |
| confirmatory-v1 | Frozen final method/configuration comparison | Held-out evidence under preregistered conditions |

Never port an upstream optimizer defect silently into the corrected default, and never silently fix a reference protocol while claiming exact reproduction. Record both the reference behavior and the corrected choice.

## 3. Core method matrix

The matched comparison should include, subject to the registered stage budget:

1. Random and simple reactive controls.
2. Frontier assignment with a functioning planner.
3. An uncertainty-aware frontier/revisit heuristic.
4. Recurrent MAPPO with a conventional classical critic.
5. A compressed split classical critic using the same latent dimension as the quantum comparison.
6. A classical bilinear or graph interaction critic.
7. A split quantum critic with no inter-agent input entanglement.
8. eQMARL with the same split architecture and enabled input entanglement.
9. qfCTDE as a separate centralized-circuit topology comparison.

Not every heuristic needs a learned critic. Resource and task definitions must still match. If a heuristic uses privileged information, label it as an oracle/reference and do not compare it as though it has the same input contract.

The broad matrix is for development and attribution. The Stage 08 confirmatory study selects at most four learned methods before final testing: a strong classical method, the matched split quantum no-entanglement control, and eQMARL are required. A fourth method is optional and selected using validation evidence. Claims concerning another method must use its own clearly identified exploratory evidence rather than imply it participated in the final study.

Keep the common estimator, actor architecture, planner, candidate generation, observation preprocessing, reward, and evaluator fixed within a critic comparison. A fixed-encoder diagnostic is a separate useful ablation; sharing an architecture does not mean independently trained models must have identical learned weights.

Equal tuning effort means comparable trial budgets and validation opportunities. Different parameter groups may use different learning rates. Parameter count, wall time, and environment sample budgets cannot always be matched simultaneously; identify which constraint defines each comparison.

For input-entanglement attribution, disable only the initial inter-agent entanglement in the split circuit. Keep local gates, readout, encoding, trainability, dimensions, initialization policy, optimizer treatment, and sampling settings comparable. qfCTDE does not satisfy this control.

## 4. Task and information specification

Each resolved environment configuration must declare:

- Map dimensions, units, generation rules, and solvability assumptions.
- Robot count, kinematics, collision handling, and initial priors.
- Sensor models, noise, field of view, range, and landmark association.
- Local estimator backend and relevant inference settings.
- Actor observation fields and history.
- Critic information fields during training.
- Communication payloads, timing, latency, loss, and costs.
- Action candidates, masks, low-level control, goal duration, and time budget.
- Rewards, metric definitions, success thresholds, and termination rules.

### Information conditions

Maintain distinct named conditions:

- **local-only:** only local measurements/history and allowed direct sensing of the environment.
- **message-limited:** local information plus delivered packets under a declared policy.
- **shared-map:** a fused map is available; exchange and processing costs are recorded.
- **oracle:** extra ground-truth or complete information is intentionally supplied for diagnostic upper/reference behavior.

No global map is free merely because it is cropped into a local patch. An information-isolation test must hold an agent's own belief and delivered messages fixed, modify another agent's private information, and verify unchanged actor input in the local-only condition.

Ground truth may generate sensors, training rewards where explicitly allowed, and evaluation metrics. It must not leak into decentralized actor features or candidate selection. Reward privilege is disclosed and shared by compared methods.

### SLAM assumptions

Start with known landmark identities and unknown landmark positions. Declare the initial reference/gauge explicitly. Use noisy body-frame motion and a real observation model. Keep current-pose error, trajectory error, and map error distinct.

When factors or estimates are exchanged, record identities and avoid counting duplicate information as independent. If unknown cross-correlations are approximated, disclose the approximation. A diagonal covariance heuristic driven by newly observed cells is an exploration proxy, not a calibrated SLAM posterior.

## 5. Data splits and seed streams

Create manifests for training, development, validation, and final test scenarios before their respective use. Use disjoint explicit scenario IDs or deterministic keyed generation; do not rely on overlapping seed-plus-episode sequences.

Maintain separate streams for:

- Model/actor initialization.
- Critic initialization.
- Training map generation.
- Evaluation maps.
- Motion/sensor noise.
- Policy action sampling.
- Quantum measurement shots.
- Communication drop/delay draws.

Pair map/scenario conditions across methods. Where appropriate, use keyed exogenous noise indexed by scenario, robot, sensor, and time so implementation-specific random-number consumption does not accidentally alter the comparison.

A training seed represents an independent trained model. Hundreds of episodes from one model do not become hundreds of independent training replications.

Development and validation data may guide diagnostics and selection within the registered budget. Final test scenarios must not guide architecture, hyperparameter, reward, or checkpoint selection.

## 6. Proposed bounded resource profiles

The implementation must materialize the selected profile in a resolved configuration before running it.

| Profile | Proposed initial size | Intended use |
| --- | --- | --- |
| smoke | 1 seed; at most 2 short episodes of 50 primitive steps; only the minimum updates needed to exercise the path | Installation, serialization, and end-to-end correctness |
| diagnostic | Small fixed datasets and scripted trajectories; at most 1 hour per diagnostic family by default | Circuit parity, estimator behavior, value fitting, and leakage checks |
| tuning | At most 6 predefined configurations per learned method, each with 2 development seeds and 10,000 primitive steps per seed initially; development/validation only | Comparable bounded optimizer/capacity selection |
| pilot | 3 training seeds; 20,000 primitive steps per seed initially; 30 validation episodes per policy | Detect collapse, measure variance, estimate effect and cost |
| confirmatory | Target 8 training seeds; 100,000 primitive steps per seed initially; 100 paired held-out episodes per policy; at most 4 selected learned methods | Locked final comparison |

Proposed operational caps:

- Use the existing available CPU resources by default; this protocol does not authorize purchasing or provisioning new services. Any later accelerator profile must identify already authorized available resources explicitly.
- Start with one training job at a time. Increase concurrency only after measuring memory and contention.
- Use at most the available resources and initially no more than 8 CPU worker threads per job.
- Stage wall-time cap: 24 cumulative active execution wall-hours, aggregated across its runs, attempts, and resumed sessions.
- Smoke attempt: 15 minutes.
- Pilot run: 2 hours.
- Respect a stricter host/job allocation whenever one exists.

Budget usage is cumulative. Checkpoint/resume, retries, renaming experiments, or creating another branch do not reset it. Stage wall usage sums the wall durations of its recorded execution segments, including diagnostics, attempts, and resumes; concurrent job segments each contribute their duration. Idle time between user sessions is not an active execution charge, but running jobs continue consuming their registered budget.

Record CPU core-hours as aggregate process CPU time across the job and its workers, and record GPU device-hours separately as allocated device count multiplied by execution duration. Record measured utilization separately where available. These are distinct resource counters, not fungible aggregate “compute-hours”; an unused GPU counter is zero. Use the most restrictive applicable wall-time or host allocation cap.

These caps deliberately require a throughput preflight. If the initial step/seed/method matrix cannot fit, choose a feasible common budget or narrower preregistered question **before observing comparative outcomes**. Preserve the classical and matched no-entanglement controls. Do not silently shorten only slow or losing methods and retain a sample-matched claim.

A wall-time-matched comparison may be registered separately from a sample-matched one. If the final target of eight seeds is infeasible, declare the smaller count before the final experiment and report its limitations; do not claim the target count was completed.

At most one pilot-budget extension is allowed by default, supported by a documented learning-curve or throughput diagnosis. Further work requires a new explicit budget record. Unavailable compute leads to a bounded or partial result, not invented completion.

## 7. Required run package

Store one immutable directory per run:

~~~text
experiments/<stage>/<run_id>/
  manifest.json
  resolved_config.yaml
  metrics.jsonl
  summary.json
  diagnostics.json
  artifact_manifest.json
  report.md
~~~

File contents:

- **manifest.json:** identity, status, source/config/data hashes, seeds, runtime, budgets, and execution segments.
- **resolved_config.yaml:** all effective values, including defaults and command overrides.
- **metrics.jsonl:** raw episode/update records, including failures and partial progress.
- **summary.json:** computed aggregates with analysis version and input hashes.
- **diagnostics.json:** numerical checks, model/gradient statistics, and gate outcomes.
- **artifact_manifest.json:** files/checkpoints, byte sizes, hashes, and verified GitHub artifact locations when external to the Git tree.
- **report.md:** hypothesis, method, results, limitations, gate decision, and next action.

The [run-manifest schema](../schemas/run_manifest.schema.json) is authoritative for field names, required fields, statuses, types, and schema version. Stage 01 must implement schema validation plus cross-file identity and artifact checks; schema validity alone does not prove execution. Expand the schema explicitly when needed rather than creating incompatible manifest conventions.

A planned stage/run index may contain unresolved null identities, clearly labeled as planning records. Create an actual execution manifest only after the source commit, clean tree, configuration hash, applicable split manifests, and effective seed streams are resolved. Unused streams, splits, and quantum fields follow the schema's explicit null conventions; a required source/config identity cannot be null. Do not create fake completed or partially populated executed runs to illustrate the format.

Record dependency versions, hardware/device, dtype, gradient method, quantum backend, trainable parameter counts, observable definitions, shots, noise settings, optimizer groups, normalization, and environment configuration. Do not dump the entire process environment or credentials.

## 8. Metrics and diagnostics

### Task outcomes

Report at least:

- Physical sensor coverage using simulator truth for evaluation.
- Estimated-map coverage as a separate quantity.
- Occupancy or landmark-map accuracy over a fixed evaluation domain.
- Current/final pose error.
- Trajectory ATE/RPE or a precisely defined alternative.
- Travel distance and elapsed primitive steps.
- Collisions and localization failures.
- Coverage–accuracy success according to preregistered thresholds.
- Redundant exploration and inter-robot information exchange.

Do not call instantaneous positional RMSE across robots a trajectory-wide ATE. Do not compute apparent map accuracy only on an outcome-dependent favorable region without also reporting fixed-domain coverage and error.

### Learning outcomes

- Return and reward components.
- Policy entropy and action/goal frequencies.
- Value/target means, variances, sign ranges, and value-fit error.
- Advantage distribution.
- Encoder, circuit, actor, and critic gradient norms.
- Primitive steps, high-level decisions, updates, and wall time.
- Checkpoint identity and selection rule.

### Quantum and communication resources

- Total and per-agent qubits.
- Logical depth and gate counts with a declared convention.
- Measured observables and grouping.
- Actual circuit evaluations and shots.
- Gradient evaluation method and repeated executions.
- Noise model and any retry/discard/loss behavior.
- Classical messages for SLAM, critic training, and execution separately.
- Quantum distribution/return operations in a claimed distributed model.

An exact-expectation simulation must say shots are not physically sampled; it cannot report zero shots as a practical hardware cost. Extra classical unary-value messages in a hybrid critic also count.

## 9. Reward, horizon, and estimator checks

Reward components should be individually logged. A candidate decomposition is progress plus meaningful uncertainty reduction minus travel, collision, and communication costs. The exact coefficients must be committed before comparison.

For uncertainty rewards:

- Compare a fixed-dimensional, gauge-consistent marginal or an explicitly dimension-aware quantity.
- Permit information gain from known landmarks and loop closures. Compare the measurement posterior with the same-step prediction/no-measurement control, not an earlier timestamp with less accumulated process noise.
- Avoid treating correlated/repeated measurements as fresh independent evidence.
- Check that a motion/sensing cycle cannot create unlimited reward through a bookkeeping artifact.
- Do not allow estimated pose drift to manufacture physical coverage.

If using potential-based shaping, specify the state/belief, discount, and terminal convention rather than assuming policy invariance for an arbitrary reward.

For finite-horizon tasks, supply remaining budget and apply true terminal semantics. For an external rollout cutoff, bootstrap appropriately. Reset recurrent state only at the defined episode boundaries. Variable-duration high-level actions require accumulated rewards and duration-consistent discounting.

## 10. Statistical analysis

Prespecify primary outcomes and comparisons. A useful primary comparison is eQMARL versus its matched no-entanglement split critic, accompanied by the strongest relevant classical control.

For each trained seed, aggregate its test episodes into a seed-level outcome. Report the distribution across training seeds and paired differences under shared test conditions. Hierarchical bootstrap or seed-level intervals can be used if their assumptions and resampling units are stated.

Report effect sizes and uncertainty, not only p-values. Failure to detect a difference is not evidence of equivalence. An equivalence analysis requires a prespecified practical margin.

Keep exploratory analyses labeled exploratory. If many methods/conditions are compared, distinguish primary contrasts from secondary sweeps and account for multiplicity where inferential claims require it.

Failed and budget-stopped runs remain in the ledger. Describe whether their outcomes represent algorithmic failure, infrastructure failure, or censoring. Do not remove an inconvenient seed without a reproducible, method-independent reason.

Report sampled and greedy policies separately if both are used. An animation of a single rollout is explanatory material, not another independent test.

## 11. Publication and Git identity

### Required sequence

1. Commit implementation and the resolved experiment definition as **Commit A**, publish A on the stage branch, and verify its remote SHA before starting a reportable run.
2. Run using A as source_commit.
3. Write metrics, diagnostics, analysis, and artifact manifests.
4. Commit those outputs as **Commit B** and push the stage branch.
5. Read the remote branch/files and verify B.
6. Record B as results_commit in a subsequent publication ledger update or another immutable external record.

Do not put a commit's own SHA inside that same commit's content; that is self-referential. The source commit identifies code used for training, while the results commit identifies the published evidence. A later ledger commit can record both.

Every stage report must link its exact source and result commits, protocol, run IDs, gate outcome, and unresolved limitations. Keep a stage PR as an inspectable summary. Do not delete stage branches after integration.

### Artifact storage

- Put configs, raw tabular metrics, summaries, diagnostics, analysis scripts, and reports in the repository.
- Retain final and selected-best checkpoints for completed runs; retain resumable checkpoints for interrupted work until resolved.
- Small checkpoint files may be committed directly within a declared size budget.
- Proposed initial direct-Git limits: 10 MiB per binary file and 100 MiB of new binary artifacts per stage.
- Larger required artifacts should use verified Git LFS or immutable GitHub Release assets, with SHA-256 hashes and exact references in artifact_manifest.json.
- Do not rely solely on temporary Actions artifacts, an external dashboard, or a manifest pointing to files that were never uploaded.
- If artifact publication is unavailable, push the evidence that can be published and record blocked_publication. Do not say every stage artifact is archived until upload verification succeeds.

A failed or negative experiment gets the same evidence treatment as a successful one. Large redundant intermediate checkpoints can follow a declared retention policy, but no retained result may depend on an unavailable unrecorded file.

## 12. Failure handling and restart

### Stage outcomes and run statuses

The following outcomes are descriptive stage-ledger/gate states. Execution-manifest `status` values use the schema's vocabulary: `planned`, `running`, `completed`, `failed`, `stopped_budget`, `abandoned`, `interrupted`, or `invalidated`. Store more specific diagnoses in `failure`, and publication status in `publication`. A valid null experiment is a `completed` run, with its null finding recorded in the report; a resumable interruption records `interrupted` with an interruption/resume diagnosis until resumed. Invalidated historical results retain their files and record `invalidated` with the reason. A dependency blocked before execution belongs in the planning/stage ledger rather than a fabricated execution manifest.

| Status | Meaning |
| --- | --- |
| planned | Defined but not executed |
| running | Active work with a resolved identity and budget |
| passed | Correctness and required evidence gates passed |
| completed_null_result | Valid experiment with no claimed benefit |
| failed_correctness | An implementation or semantic gate failed |
| blocked_dependency | Required capability/runtime unavailable |
| blocked_compute | Authorized resource budget or execution limit prevents completion |
| blocked_publication | Required results/artifacts are not yet fully available on GitHub |
| stopped_budget | A particular run reached its registered cap |
| interrupted | A resumable run stopped unexpectedly |

Stage and run statuses are distinct. A completed stage can contain failed runs if the failures were investigated and the required study is valid.

### Remediation bounds

- At most two diagnosis/patch cycles for the same unexplained correctness failure by default.
- Diagnose the smallest reproducer before retraining.
- Commit fixes and use a new source identity for a fresh run.
- Resume only when checkpoint, configuration, optimizer, normalizer, RNG states, and data identity are compatible.
- If a correctness fix invalidates an earlier result, mark it invalidated and retain it; do not silently overwrite it.
- Never advance a performance claim past an unresolved correctness gate.
- Continue unaffected documentation, analysis, artifact recovery, or classical checks when another component is blocked.

### Resume checklist

At session restart:

1. Read AGENTS.md, the execution plan, and the latest stage/run records.
2. Verify remote branch heads and local changes.
3. Locate and hash input artifacts/checkpoints.
4. Identify whether prior processes are still running; do not launch duplicates.
5. Confirm remaining resource budget.
6. Restore complete state for a compatible resume, or create a clearly distinguished fresh attempt.
7. Record the next action and its expected output.

Checkpoint state should include model, optimizer, normalization statistics, scheduler, RNG states, progress counters, and any environment/recurrent state required for exact continuation. If exact continuation is unsupported, say so and restart only at a documented boundary.

## 13. Completion and claims

A result is ready to report when:

- Its task and information contracts are clear.
- Its implementation gates passed.
- Its comparison and tuning budgets are honest.
- Its raw outcomes and failures are retained.
- Its source/config/data identities resolve.
- Its analysis can be regenerated.
- Its required artifacts are verifiably available.
- Its conclusion matches its statistical and resource evidence.

Permitted conclusions include a useful quantum benefit, a conditional benefit, no detected benefit, a stronger classical interaction model, a resource-limited result, or an identified failure mechanism.

The project does not require a particular winner. It requires a reproducible answer to the registered question.
