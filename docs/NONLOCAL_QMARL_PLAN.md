# Learned nonlocal coordination: prospective QMARL study

## Scientific question

Can a decentralized quantum multi-agent policy, trained from return rather than assigned the analytic solution, exceed the best achievable classical no-communication policy on a cooperative task with private observations?

The benchmark is the CHSH cooperative contextual bandit. Agent A observes `x in {0,1}`, agent B observes `y in {0,1}`, and neither receives the other observation. They return binary actions `a,b` and receive common reward `1[a xor b = x and y]`. Inputs are uniform. Classical decentralized policies, including shared randomness independent of the inputs, have expected reward at most `0.75`. A Bell pair with local observation-conditioned measurements can attain `cos(pi/8)^2 = 0.853553...` without communication.

This is a new protocol family. It does not alter the active-SLAM results and it cannot turn the earlier critic null into a positive result. Its possible contribution is a scoped learned-policy quantum advantage in decentralized coordination, plus a reproducible finite-shot/noise frontier.

## Integrity review of the prior study

Before the new claim, Stage 15 reruns the prior unit suite and reconstructs the Stage 13 estimand independently from archived episode rows. It checks seed pairing, metric extraction, checkpoint hashes, entanglement-only parameter matching, and the analysis resampling unit. A clean review supports the recorded result only within its stated scope. Any discrepancy that can change the primary result blocks the new study until it is reported and repaired transparently.

## Information and intervention contract

- Each actor receives only its own input bit.
- There is no classical communication after inputs are issued.
- Shared randomness may be sampled before inputs and is explicitly allowed for the strongest classical control.
- The entangled policy receives one Bell pair per decision and applies only a local, observation-conditioned single-qubit rotation before local computational-basis measurement.
- The separable quantum control uses the identical trainable local measurements and optimizer without Bell preparation.
- A centralized classical oracle is a diagnostic upper bound and is never an admissible comparator for the no-communication claim.
- Simulator access to both inputs is limited to reward calculation, audit, and exact expectation evaluation.

## Methods

1. `classical_local`: independently parameterized stochastic local policies.
2. `classical_shared`: a trainable convex mixture of deterministic local strategies using pre-input shared randomness.
3. `quantum_separable`: matched local quantum measurement policy initialized without entanglement.
4. `quantum_entangled`: the same policy with Bell preparation.
5. `central_oracle`: diagnostic policy with both input bits; excluded from inference.

The classical shared-randomness control is accompanied by exhaustive enumeration of deterministic local strategies. This certifies the 0.75 classical ceiling rather than relying on optimizer failure. Parameter counts, optimizer opportunities, random starts, and stopping rules are recorded. The physical resource table reports Bell pairs, qubits, circuit executions, and shots.

## Outcomes and inference

The primary outcome is each independently trained seed's exact expected win probability under uniform inputs, evaluated from the frozen final policy. The primary hypothesis is accepted only if both prospectively locked conditions hold:

1. the two-sided paired-seed 95% bootstrap interval for `quantum_entangled - max(classical_shared, quantum_separable)` excludes zero on the positive side; and
2. the one-sided 95% seed-level lower confidence bound for `quantum_entangled` exceeds the certified classical ceiling `0.75`.

The minimum effect of scientific interest over the ceiling is `0.05`. Report the full interval even if the gates pass. A finite-shot test uses fresh locked RNG streams and reports a binomial interval; it supports operational reproducibility but does not replace seed-level inference. Secondary outcomes are convergence updates, failure rate, seed dispersion, and performance under registered local depolarizing noise.

No result from tuning or pilot seeds enters confirmatory inference. Noisy conditions are secondary unless Stage 17 freezes a single robustness threshold before Stage 18. Multiple exploratory noise points are labeled descriptive.

## Compute plan

Every stage has a 24 cumulative active wall-hour cap, one training job at a time, at most eight CPU threads, and no paid compute. The implementation is limited to two qubits. Stage 16 benchmarks exact and finite-shot throughput before later budgets are frozen.

Stage 17 may test at most four optimizer configurations per learned method, two development seeds each, followed by four pilot seeds per selected configuration. Stage 18 targets 16 fresh paired training seeds for the four inferential methods. The initial limit is 2,000 optimizer updates per seed with final-checkpoint selection. The benchmark can reduce this limit prospectively if convergence and precision gates are met; it cannot increase it after confirmatory outcomes are opened. Exact evaluation uses four input contexts. Finite-shot evaluation is capped at 8,192 decisions per trained seed. Noise diagnostics are capped at three prespecified probabilities.

## Stages and gates

### Stage 15 — plan and integrity audit

Deliver this prospective plan, independent prior-result audit, stage registry, budgets, and claim boundaries. Gate: no unresolved discrepancy capable of changing the Stage 13 primary result; new hypotheses and inference rules are committed before implementation or outcomes.

### Stage 16 — benchmark and correctness

Implement the environment, policy families, exact probabilities, sampled execution, artifact schema, and resource counters. Required gates: exhaustive classical strategies max at exactly 0.75; analytic Bell strategy equals `cos(pi/8)^2` within `1e-8`; input isolation holds; separable policies cannot exceed 0.75 under exhaustive/global optimization; sampled means agree with exact probabilities within registered binomial tolerance; gradients agree with finite differences; serialization/replay is exact.

### Stage 17 — bounded development and freeze

Use development-only seeds for equal-budget optimizer selection. Diagnose gradients, convergence, boundary violations, and throughput. Freeze methods, hyperparameters, seed streams, update cap, exact evaluation, finite-shot manifest, analysis code, and resource estimate. Gate: every inferential method has a functioning training path; the quantum entangled pilot must show the experiment is capable of testing the primary gate, without using confirmatory seeds. Failure passes forward only as a documented infeasible mechanism and cancels Stage 18 advantage inference.

### Stage 18 — locked confirmation

Run all registered fresh seeds and retain failures. Evaluate frozen final policies exactly and with locked finite shots. Gate: every planned run has a recorded status, all tables trace to raw rows, and the preregistered analysis runs without model or hyperparameter changes. Positive, null, and negative outcomes all complete the scientific stage; only the two-part primary gate permits the phrase `quantum advantage`.

### Stage 19 — release

Recreate the analysis from archived raw data, replay representative checkpoints, verify hashes and remote retrieval, and publish a paper-ready report, tables, limitations, and reproduction commands. Gate: a clean session reproduces the primary numbers and all claims match the registered decision rules.

## Claim boundaries

A passing result establishes a learned advantage over classical decentralized no-communication policies for this two-agent nonlocal game under analytic simulation and the tested finite-shot/noise conditions. It does not establish computational speedup, hardware advantage, advantage in active SLAM, usefulness when classical communication is free, or superiority to a centralized controller. The classical 0.75 ceiling depends on the information contract; relaxing that contract changes the task.
