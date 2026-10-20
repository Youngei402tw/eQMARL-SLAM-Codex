# Reference audit and the differences this project will test

**Audit date:** 2026-10-20.  
**Evidence type:** inspection of pinned repository source and the repositories' reported results.  
**Execution status:** no new training run, environment installation, circuit parity run, or independent replication was performed for this planning document.

## 1. Sources and terminology

“OLD” below means the current inspected revision of the repository named `eQMARL-SLAM`. It does **not** mean that every problem in that repository's historical change log remains present. “NEW main” and “NEW active” identify two distinct revisions of `eQMARL-SLAM-NEW`.

| Reference | Inspected revision | Role |
| --- | --- | --- |
| [Original eQMARL paper](https://arxiv.org/abs/2405.17486) | arXiv:2405.17486; archive the selected PDF/version in the execution manifest | Scientific method reference |
| [Original eQMARL code](https://github.com/news-vt/eqmarl/tree/6b1b2e3817f661da81aad5e7188a362380dbdd1d) | `6b1b2e3817f661da81aad5e7188a362380dbdd1d` | TensorFlow/TFQ/Cirq circuit and learner reference |
| [OLD](https://github.com/Youngei402tw/eQMARL-SLAM/tree/60388c4641a6ce23b4b750f6bde4bc97b7621036) | `60388c4641a6ce23b4b750f6bde4bc97b7621036` | Corrected MiniGrid-style cooperative mapping implementation |
| [NEW main](https://github.com/Youngei402tw/eQMARL-SLAM-NEW/tree/d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f) | `d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f` | Preserved faithful active-SLAM pilot and its handoff |
| [NEW active](https://github.com/Youngei402tw/eQMARL-SLAM-NEW/tree/4ff242a69f33ac76a3127b6a960552f6b9a629d5) | `4ff242a69f33ac76a3127b6a960552f6b9a629d5` | `active-slam-improvements`: bounded-pose option, milestone protocol, analysis, and reported results |

The NEW handoff identifies `62a99cf3f4786644980d7c36e5d8052ce338b94a` as the protected pilot's implementation commit. Its main-branch handoff is a later documentation snapshot. Use the implementation SHA for a faithful pilot rerun and preserve the corresponding configurations and artifact hashes; do not substitute the newer milestone protocol. [NEW handoff][N-H]

The original repository declares CC BY 4.0 in its license and package metadata. Future adaptations must retain attribution and record modifications. This new planning stage contains an audit and implementation decisions, with no copied upstream implementation. [Original license][R-L] · [Original metadata][R-P]

## 2. What is already useful and what is already fixed

The user's implementations contain substantial work that should be retained as reference evidence. Rewriting them in PyTorch would not by itself fix the scientific comparison.

| Topic | Current source evidence | Consequence for this project |
| --- | --- | --- |
| OLD observations and dynamics | Per-agent known maps; explicit unknown/free/wall/FOV/visited channels; synchronous movement resolution; own position and heading in auxiliary input | Preserve the local information and transition invariants as regression fixtures. Do not repeat the claim that OLD currently feeds the entire hidden map to the actor. [OLD core][O-C] |
| OLD learner corrections | Detached GAE targets and advantages, entropy subtracted from the minimized objective, mean losses, optional advantage normalization and gradient clipping | These historical defects have already been addressed in the inspected source. Port semantics deliberately rather than advertise them as new fixes. [OLD learner][O-A] |
| OLD baseline distinction and reproducibility | Separate sCTDE/fCTDE builders, explicit model and map seeds, evaluation routines, numerical sampling safeguards | Reuse the comparison ideas and regression cases. Confirm runtime behavior in the new stack before claiming replication. [OLD models][O-M] · [OLD runner][O-R] · [OLD numerical helpers][O-N] |
| NEW reward and optimization | TD targets detached, entropy sign corrected, mean team-reward option, advantage normalization/clipping, and uncertainty reward prevented from becoming positive when no new cells are observed | These fixes are already present; the new protocol must identify its additional changes. Verify resolved settings because implementation defaults and experiment YAML choices differ. [NEW learner][N-A] · [NEW environment][N-E] · [NEW handoff][N-H] |
| NEW information shape | Faithful critics receive the two 147-feature observations rather than a duplicated global descriptor | This corrects the explicit critic-input duplication. The observation's provenance still needs separate review. [NEW models][N-M] · [NEW environment][N-E] |
| NEW active engineering | Explicit estimator module, optional pose bounds, boundary checks, milestone validation, reward diagnostics, protocol-specific analysis and tests | Preserve these as a separate improved protocol. A bounded estimate or changed reward is not a like-for-like rerun of the pilot. [Active backend][A-B] · [Active environment][A-E] · [Active analysis][A-A] |

The OLD validation report explicitly distinguishes executed checks from dependency-gated TFQ training checks. Its documented tests are not proof that the current new PyTorch pipeline or a full legacy TFQ learning run has executed. [OLD validation report][O-V]

## 3. Mapping, active exploration, and SLAM are different benchmarks

### OLD: a useful known-localization mapping control

OLD constructs each agent's own map from its field of view, while its auxiliary observation contains normalized true grid position and one-hot heading. It is therefore a cooperative mapping/exploration task with supplied localization. That is a useful first environment for studying information boundaries, task allocation, rewards, and learning. It cannot establish improved pose estimation or loop closure. [OLD core][O-C] · [OLD validation report][O-V]

Keep this task class as an early control. It separates a policy failure from an estimator failure. MiniGrid-style 2D environments are appropriate here; adopting a full SLAM stack before these checks would add another source of error.

### NEW: estimated pose and map, with heuristic uncertainty

NEW feeds noisy odometry and LiDAR into a scan-matching grid backend. In the active branch, scan matching evaluates a small local pose search and updates occupancy log odds. Its covariance update adds hand-set diagonal process noise, shrinks covariance according to a scan score and newly observed cells, and clips diagonal entries. This is a transparent uncertainty proxy, not a calibrated joint posterior from the measurement likelihood. The optional pose bounds are an additional prior/constraint. [Active backend][A-B]

The new research environment should have an estimator contract and an explicit 2D measurement model. An anchored GTSAM factor graph is a reasonable later backend for pose/landmark estimation; a validated EKF is another separately named option. Test measurement residuals, covariance consistency, gauge handling, coordinate conventions, and association assumptions. A library's presence alone does not validate uncertainty.

Do not carry over “only new cells may reduce uncertainty” as a universal SLAM law. It closes a reward loophole in the heuristic backend, but revisiting a known landmark or closing a loop can legitimately reduce uncertainty in an estimator. For actual belief-based active SLAM, verify that information rewards follow valid belief updates and cannot be repeatedly collected from the same information without justification. This is the reason to separate a coverage-only control from a belief-based task.

## 4. Observation provenance and communication assumptions

In both inspected NEW revisions, `_observation` first calls `_fuse_beliefs` on **both** robots' maps, builds occupancy/observed/frontier channels from the fused result, and then crops a patch around each robot's estimated pose. The returned critic input is a copy of those actor observations. Thus a local crop is spatially local but may contain information contributed by the other robot. This is not direct exposure of the hidden ground-truth map. It is globally available map sharing and an assumed common map frame. [NEW environment][N-E] · [Active environment][A-E]

This behavior is legitimate for a shared-map baseline. It cannot support a claim that execution uses only each robot's own private sensing without communication. Removing a duplicated global descriptor from the critic did not remove this separate information path.

The new environment must register explicit regimes: private local belief; constrained messages with payload/range/rate/delay/loss; and a shared-map baseline. Counterfactual observation checks should change another robot's private history while keeping the first robot's sensor history fixed. In the private regime, the first actor's input must remain unchanged. In message-enabled regimes, it may change only through recorded messages. Keep evaluator truth and reward diagnostics outside actor inputs. If training uses privileged information, label that critic regime and apply it consistently across methods.

## 5. Circuit differences matter more than the repository names

### The original and NEW faithful partite circuits

The original partite builder prepares an initial entangled state across corresponding partition qubits, then applies variational, encoding, and entangling operations within each partition. NEW's faithful active-SLAM model uses this family with a single product-of-Z observable over all qubits. Its qfCTDE builder uses a global circuit and is a distinct architecture. Consequently, qfCTDE is not the matched “turn off initial entanglement” control for eQMARL. [Original circuits][R-C] · [NEW models][N-M]

The NEW active-SLAM eQMARL builder explicitly passes `input_entanglement=True`. The underlying layer exposes a toggle, so the matched ablation is implementable, but it needs an explicit model/configuration path and a recorded test. Freeze all other gates, feature encoders, observables, parameter initialization policy, optimizers, and training/evaluation budgets for that ablation. [NEW models][N-M] · [NEW layers][N-L]

### OLD is a different circuit variant

The current OLD `build_partite_variational_circuit` adds cross-agent CZ gates after the local blocks in **every** layer. It also uses a different rotation/encoding order from the original. Its readout constructs per-partition products of X, Y, and Z, followed by a nonlinear classical head and an unconstrained output. Those choices are experiments worth considering, but they do not preserve the original initial-entanglement-followed-by-local-processing architecture. Source code at the selected SHA is authoritative if a historical change-log description differs. [OLD quantum code][O-Q] · [OLD models][O-M]

### Why local-only Bell readout is a required diagnostic

With `d` Bell pairs between two `d`-qubit partitions, each partition's reduced state is maximally mixed. After arbitrary local unitaries it remains maximally mixed. All local traceless Pauli observables then have exact expectation zero, independent of the local data or parameters. Joint correlators can remain informative. This is a direct reduced-state calculation; it is not a training hypothesis.

This warning does **not** prove OLD's current readout has zero signal, because OLD adds later cross-agent gates, which change the assumptions. It does explain why blindly adding local observables to the original locally processed Bell architecture may add no useful features. The parity suite must check the local-zero result, nontrivial global correlations, and the changed behavior when cross-agent operations are deliberately enabled. See the tolerances and Bell fixtures in [ADR 0001](adr/0001-pytorch-stack.md).

## 6. The value-head issue is conditional and specific

NEW's faithful quantum value head implements a scaled weighted expectation of the form

$$
V(o)=\beta\,w\,\frac{1+q(o)}{2},\qquad q(o)\in[-1,1].
$$

The weight `w` is trainable and unconstrained. It can become negative, so it is incorrect to describe this head as always nonnegative. With fixed parameters and positive beta, however, the single-observable output has one sign across inputs: its offset and slope are tied. That limits the head if the intended value targets need both positive and negative values at the same point in training. Whether it is materially harmful must be checked against actual target distributions and fitted-value diagnostics. [NEW readout][N-M] · [NEW weighting layer][N-L]

A separate corrected variant can use `V=b+sum_j(w_j*q_j)`, with a free bias and signed coefficients. First retain the reference head for parity; then compare the affine head on identical data and learner settings. Apply compatible output freedom to the matched quantum controls. OLD's current unconstrained dense output already has a different form, so this diagnosis must not be applied indiscriminately to all three implementations. [OLD models][O-M]

Likewise, `trainable_w_enc=False` in NEW freezes the quantum layer's encoding-scale weights. It does **not** freeze the preceding `LocallyConnected1D` classical feature encoder, whose weights remain trainable. Log parameter groups and gradient paths directly instead of interpreting that flag as “the entire encoder is frozen.” [NEW models][N-M] · [NEW layers][N-L]

## 7. Learner semantics and a stronger classical comparison

NEW's batch update combines `terminated or truncated` when deciding whether to bootstrap. If the time limit is an external cutoff of a continuing task, this suppresses a valid bootstrap. If the intended objective is genuinely finite-horizon, zero bootstrap at its boundary may be appropriate, but that horizon must be part of the task specification and observations/history must support the intended value function. Treat this as a protocol decision to test, not an unconditional claim that every time-limit implementation is wrong. [NEW learner][N-A] · [Gymnasium termination/truncation contract](https://gymnasium.farama.org/introduction/basic_usage/)

First match the original selected MAA2C/A2C behavior on fixed trajectories. Then introduce corrected masks and a common recurrent PPO/GAE implementation as named changes. Shared actor weights must retain separate hidden states for each robot. Verify agent/batch axes, detached targets, entropy sign, clipping, rollout/reset boundaries, and action duration discounting. The official [MAPPO implementation](https://github.com/marlbenchmark/on-policy) is an algorithmic reference; use a tested modern dependency set instead of its legacy installation pins.

Use validation-only, equally budgeted tuning for classical and quantum critics. Record both parameter-matched classical controls and strong practical classical baselines. Match observation access and the classical feature bottleneck when testing the quantum representation. Also include a useful nonlearning exploration baseline, such as frontier selection with conflict resolution. Larger classical models are not automatically unfair, and a smaller quantum parameter count is not a compute-cost measurement.

## 8. What the existing results support

The active branch reports five training seeds for the 99%-coverage protocol. Its final-100 training-window means are 0.9808 coverage for eQMARL and 0.9824 for qfCTDE; the reported paired coverage difference is -0.0016 with a 95% t-interval half-width of 0.0094. The document reports much weaker coverage for its current classical configurations. These are **repository-reported results, not rerun or independently recomputed here**. [Active results summary][A-R]

The appropriate conclusion is that the reported comparison does not resolve an eQMARL-over-qfCTDE advantage. An interval containing zero is not a formal equivalence result. The source explicitly says the statistics come from online training episodes, not frozen-policy evaluation on a reserved map set, and that optimizer settings and parameter counts differ. Those limitations prevent attributing the observed classical/quantum gap specifically to entanglement. [Active results summary][A-R]

The source handoff also separates the 90%-coverage faithful pilot from the changed 99% milestone protocol. Their rewards and success definitions differ, so pooling their returns or comparing success percentages as a single benchmark would be misleading. [NEW handoff][N-H] · [Active protocol][A-P]

## 9. Unified evaluation and evidence requirements

Every new stage must push its configuration, code revision, dependency manifest, completed/failed status, raw episode records, seed-level analysis, and report to its own GitHub branch or branch-linked artifact location. A checksum without an accessible artifact is an integrity record, not a backup. Store enough data to recompute the report without rerunning training.

The evaluator should implement the following common contract:

1. Freeze policy parameters and preprocessing statistics; reserve disjoint training, validation, and final-test map/noise seeds. Pair methods on test maps and sensor randomness using separately keyed streams so action-dependent RNG consumption does not silently break pairing.
2. Record independent training seeds as the top-level statistical units. Repeated evaluation episodes within one trained seed improve its estimate; they do not create additional independent training runs. Report seed-level differences, confidence intervals, raw points, and failures.
3. Choose the primary metric, budget, checkpoint selection rule, comparison family, and smallest meaningful improvement before final evaluation. Use held-out results once for confirmation; a null result remains a valid outcome. A claim of practical equivalence requires a prespecified equivalence margin and an appropriate analysis.
4. Measure coverage consistently, including a physical sensing coverage measure computed from evaluation-only truth. NEW's current coverage uses its estimated observed-cell mask intersected with reachable cells; pose drift can make that differ from physically observed space. Report mapping precision/recall or IoU with a declared mask and treatment of unknown cells. [Active environment][A-E]
5. Distinguish trajectory error from an instantaneous pose error. NEW's `pose_rmse` is an RMS across the robots' current position errors, not a full trajectory ATE. The new evaluator must retain trajectories and define alignment/gauge, time aggregation, translation units, heading error, and failure handling. [Active environment][A-E]
6. Include fixed-budget coverage AUC, success, threshold time with failures accounted for, path length, collisions, uncertainty consistency, communication bytes, training/inference wall time, quantum executions/shots, and total trainable parameters. Do not report only threshold time among successful episodes as though failures had no cost.

## 10. Why the proposed approach is better supported

The proposed approach is better supported as a **research design** because it separates known-localization mapping from belief-based SLAM, audits the provenance of every policy input, verifies the quantum circuit numerically, changes one scientific factor at a time, and uses frozen-policy seed-level comparisons. Its PyTorch implementation should make those contracts easier to inspect, but the framework is not the scientific contribution.

Start in 2D. It offers enough structure to test private information, complementary observations, communication constraints, noisy estimation, exploration versus revisiting, and loop closures. Move to 3D only after a specific surviving hypothesis needs visual ambiguity, geometry, or scale that the 2D task cannot supply. Added dimensionality does not by itself favor eQMARL and can make classical feature extraction dominate the comparison.

The research claim remains open: an entangled critic may improve sample efficiency or robustness in a registered setting, tie strong classical/quantum controls, or lose after accounting for simulation/measurement cost. None of those outcomes should be predetermined by the implementation plan.

## Pinned source links

[R-C]: https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/eqmarl/circuits.py
[R-L]: https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/LICENSE.md
[R-P]: https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/pyproject.toml
[O-C]: https://github.com/Youngei402tw/eQMARL-SLAM/blob/60388c4641a6ce23b4b750f6bde4bc97b7621036/SLAM/slam_core.py
[O-A]: https://github.com/Youngei402tw/eQMARL-SLAM/blob/60388c4641a6ce23b4b750f6bde4bc97b7621036/SLAM/maa2c.py
[O-M]: https://github.com/Youngei402tw/eQMARL-SLAM/blob/60388c4641a6ce23b4b750f6bde4bc97b7621036/SLAM/slam_baselines.py
[O-Q]: https://github.com/Youngei402tw/eQMARL-SLAM/blob/60388c4641a6ce23b4b750f6bde4bc97b7621036/SLAM/quantum_layers.py
[O-R]: https://github.com/Youngei402tw/eQMARL-SLAM/blob/60388c4641a6ce23b4b750f6bde4bc97b7621036/SLAM/run_benchmark.py
[O-N]: https://github.com/Youngei402tw/eQMARL-SLAM/blob/60388c4641a6ce23b4b750f6bde4bc97b7621036/SLAM/numerics.py
[O-V]: https://github.com/Youngei402tw/eQMARL-SLAM/blob/60388c4641a6ce23b4b750f6bde4bc97b7621036/VALIDATION_REPORT.md
[N-H]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f/ACTIVE_SLAM_HANDOFF.md
[N-A]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f/eqmarl/algorithms/maa2c.py
[N-E]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f/eqmarl/environments/active_slam.py
[N-M]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f/eqmarl/active_slam_models.py
[N-L]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/d16916f42e8e39eb69dbfe84d5f2c5ecec9db30f/eqmarl/layers.py
[A-B]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/4ff242a69f33ac76a3127b6a960552f6b9a629d5/eqmarl/environments/active_slam_backend.py
[A-E]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/4ff242a69f33ac76a3127b6a960552f6b9a629d5/eqmarl/environments/active_slam.py
[A-A]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/4ff242a69f33ac76a3127b6a960552f6b9a629d5/scripts/analyze_active_slam_full.py
[A-R]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/4ff242a69f33ac76a3127b6a960552f6b9a629d5/docs/active_slam_results_summary.md
[A-P]: https://github.com/Youngei402tw/eQMARL-SLAM-NEW/blob/4ff242a69f33ac76a3127b6a960552f6b9a629d5/docs/active_slam_milestone99_protocol.md
