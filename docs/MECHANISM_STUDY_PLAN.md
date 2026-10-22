# Mechanism-focused extension: interaction phase diagram

## Status and provenance

This is a prospective extension registered after completion of Stages 00–09 and after observing the Stage 08 null result. It starts a new protocol family, uses new development/validation/test scenario manifests, and does not reinterpret or replace the original confirmatory result.

Dependency: immutable Stage 09 completion commit `2fe0e3ab73711288b705643f7ea3f12cee857fa9`. Planning branch creation base is that exact commit; the later `main` completion receipt was integrated after branch creation.

## Scientific question

Does the effect of input entanglement on learned cooperative active-SLAM policies increase when the task's return contains stronger cross-agent complementarity, after comparison with flexible classical interaction critics?

The explanatory variable is a controlled complementarity intervention in the task reward. The principal estimand is not whether eQMARL wins one condition. It is whether the paired eQMARL-minus-matched-no-entanglement effect changes between additive and high-complementarity conditions.

## Task intervention

The common two-robot 2D landmark-SLAM task, local recurrent actor, GTSAM estimator, planner, sensor model, communication contract, PPO learner, and final-checkpoint rule remain fixed. A new `belief-controller-mechanism-v1` protocol adds a belief-derived coordination objective:

- Landmarks are deterministically assigned to two complementary information groups in each scenario manifest.
- Each macro window records whether each robot produces a declared uncertainty reduction for a distinct group.
- The additive control pays the same registered per-event marginal reward without a joint bonus.
- Low and high complementarity conditions increasingly move that reward mass into a joint bonus that is paid only when both distinct contributions occur within the window.
- Reward coefficients are calibrated on scripted development trajectories before learning so conditions have comparable reward scale. Test truth is never used for calibration.
- Actor inputs contain only local belief features, local group cue, remaining budget, and delivered messages. The centralized training critic sees the same joint observations already allowed by CTDE. The task evaluator may use truth only for physical coverage, map, and trajectory metrics.

The intervention is intended to change value-function nonadditivity while preserving sensing and navigation. A scripted counterfactual table will estimate the mixed joint-action interaction

`Q(a,b) - Q(a,b0) - Q(a0,b) + Q(a0,b0)`

at fixed belief states. The measured interaction must increase monotonically across additive, low, and high conditions before policy training proceeds.

## Methods and fairness

Five critics share the Stage 08 actor, 27-dimensional observation contract, two-dimensional per-agent encoder, rollout semantics, optimizer stack, and sample budget:

1. `central_classical`: the existing strong centralized MLP.
2. `bilinear_classical`: the existing explicit bilinear interaction control.
3. `flexible_interaction`: a new symmetric classical interaction MLP over both two-dimensional latents, their product, and absolute difference.
4. `quantum_noent`: the matched quantum critic without initial inter-agent input entanglement.
5. `eqmarl`: identical to `quantum_noent` except for the registered Bell input-entanglement intervention.

Parameter counts, wall time, circuit evaluations, and samples will all be reported. The flexible classical critic is not required to have exactly the same parameter count; a small width grid is selected under the same development budget, and the selected count is disclosed. Claims concern learning under equal environment samples and tuning opportunities rather than equal wall time.

## Stages and gates

### Stage 10 — prospective plan

Publish this hypothesis, task contract, method matrix, fresh seed namespaces, analysis, resource projection, and stop/go gates before implementation or data generation.

Gate: plan validation passes; branch and source are remotely verified; no mechanism experiment has run.

### Stage 11 — controlled mechanism task

Implement the reward intervention, local cue, flexible classical critic, scripted counterfactual diagnostic, and fresh split generator. Validate information isolation, matched marginal reward scale, deterministic keyed noise, monotonic measured interaction, critic gradients, schema/archive behavior, and runtime throughput.

Smallest falsifier: at fixed scripted belief states, the high-complementarity condition fails to produce a larger absolute mixed interaction than the additive condition, or changing another robot's private state changes an actor observation without a delivered message.

Gate: all contracts pass; no final-test scenarios are accessed; projected Stage 13 matrix fits its 24-hour active-wall cap. A failed monotonicity gate stops policy comparison and is reported as a mechanism-design failure.

### Stage 12 — development and freeze

Use only development and validation scenarios. Each method receives two registered learning rates, two tuning seeds, 5,000 primitives per trial, and the same validation opportunities in the low-complementarity condition. A three-seed, 10,000-primitive pilot then runs all five methods in all three conditions. Select learning rate per method by mean validation return across the two tuning seeds; select the flexible-classical width from its equal registered grid.

Freeze the task coefficients, critic definitions, rates, seed list, checkpoint rule, and analysis. Pilot outcomes may estimate variance and feasibility but may not select the primary contrast or remove methods/conditions.

Gate: numerical and information contracts pass, all tuning/pilot outcomes are retained, no method-specific budget extension occurs, and a complete Stage 13 source/config commit is published before final-test access.

### Stage 13 — locked confirmatory phase diagram

Matrix: five methods × three complementarity conditions × eight paired training seeds. Each model receives 10,000 training primitives, 30 greedy validation episodes, and 100 paired held-out test episodes from a new immutable test manifest. One training job and one CPU thread at a time. Final checkpoint only; no outcome-based exclusion, retuning, replacement, or extension.

Primary estimand:

1. Within each paired seed, average 100 test episodes for physical coverage.
2. Compute `D_condition = coverage(eqmarl) - coverage(quantum_noent)`.
3. Compute the paired interaction contrast `D_high - D_additive`.
4. Report the mean and a two-sided 95% percentile interval from 10,000 paired-seed bootstrap resamples.

The prespecified minimum effect of scientific interest is 0.02 absolute physical coverage for the interaction contrast. The study is not an equivalence test. Secondary descriptive contrasts compare eQMARL with each classical critic in the high condition and report success, map accuracy, ATE/RPE, communication, travel, collisions, returns, action counts, gradient diagnostics, circuit evaluations, and wall/CPU cost. Secondary intervals are unadjusted and labeled descriptive.

Gate: every planned outcome is explicit, raw tables trace to hashes, frozen actor-only test execution is verified, and inference uses trained seeds as the replication unit. An interval spanning zero is a completed null result, not a reason to extend compute.

### Stage 14 — contribution and release

Rebuild the runtime from the lock, load every checkpoint, replay one validation scenario per method/condition, regenerate all analysis from raw records, audit resource and failure ledgers, and publish paper-ready tables plus a claim/limitation matrix.

Gate: fresh remote retrieval validates all artifacts and a clean session can reproduce the smoke path and analysis without training.

## Fresh data and seeds

All namespaces are disjoint from Stages 04–09:

- training maps/noise: 200000/201000 series;
- development maps/noise: 210000/211000 series;
- validation maps/noise: 220000/221000 series;
- final test maps/noise: 330000/331000 series;
- tuning actor seeds: 12001–12002;
- pilot actor seeds: 12101–12103;
- confirmatory actor seeds: 13001–13008;
- critic, action, and bootstrap streams use separately recorded offsets.

The Stage 13 test manifest is generated and hashed during Stage 11 but remains inaccessible to learning/selection code until the Stage 13 source and analysis lock are published.

## Resource envelope

No paid compute or external service is authorized. Each stage has a cumulative 24 active-wall-hour cap. Diagnostic families have a one-hour cap, each pilot/training run has a two-hour cap, one job runs at a time, and each job uses one CPU thread.

The provisional Stage 13 matrix is 120 models and 1.2 million training primitives. Stage 08 measured 24 comparable models in 4,909 experiment wall-seconds; a conservative 3× projection is 73,637 seconds before release work, below 86,400 seconds. Stage 11 must replace this projection with a measured throughput bound. If the full matrix cannot fit, reduce the common training primitive budget before any final-test access while preserving all five methods, three conditions, eight paired seeds, and 100 test maps.

## Publishable contribution and allowed conclusions

The intended contribution is a controlled interaction phase diagram connecting measurable critic nonadditivity to downstream learning, with strong classical controls and full quantum/resource accounting. A positive result may claim a scoped interaction between complementarity and the entanglement intervention. A null result may claim that the tested entanglement mechanism does not improve learning even when task complementarity increases and flexible classical critics are competitive. A negative result may show degradation or resource inefficiency.

No outcome supports a general quantum advantage, hardware speedup, untested team-size scaling, or equivalence. The earlier Stage 08 null remains reported independently.
