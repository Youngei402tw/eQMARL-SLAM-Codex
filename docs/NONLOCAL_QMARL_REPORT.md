# Where entanglement helps in QMARL: from critic nulls to a learned nonlocal-policy advantage

## Result in one sentence

A prospectively registered two-agent decentralized quantum policy learned the optimal CHSH strategy and achieved mean reward 0.853553, exceeding the certified 0.75 classical no-communication ceiling and the strongest paired trained control by 0.103554 (95% paired-seed interval [0.103553, 0.103555]); the result reproduced exactly and remained visible under registered simulated finite-shot and noise conditions.

## Why this study followed the active-SLAM null

The active-SLAM studies placed fixed Bell input entanglement inside a centralized training critic. They were internally reproducible but found no downstream coverage advantage. That design did not give decentralized actors access to a resource that changes their achievable joint behavior.

The new study therefore tested a different, prospectively declared mechanism: entanglement at decentralized execution under private observations and zero post-input communication. The CHSH contextual bandit provides a hard information-theoretic control. Each agent observes one private bit and returns one action bit. Classical local policies, even with arbitrary pre-input shared randomness, cannot exceed expected reward 0.75. A shared Bell pair permits 0.853553 without communication.

This contrast supports a concrete design principle for QMARL research: a quantum component is most credible when it changes the feasible decentralized correlation set under an explicit information contract. Adding an entangled feature map to a centralized critic does not by itself create that resource advantage. The two task families differ, so the experiments do not isolate component placement as the only causal factor.

## Prospective protocol

Stages 15–19 were separated into plan/audit, correctness, development, confirmation, and release. The prior Stage 13 result was independently reconstructed from all 12,000 raw held-out episodes before the new experiment. The new plan, controls, compute caps, classical ceiling, meaningful-effect threshold, and claim boundaries were published before implementation outcomes.

The confirmatory study used four policies:

- independent classical stochastic local actors;
- a classical shared-randomness policy represented as a convex mixture of all 16 deterministic local strategy pairs;
- a separable quantum policy with the same four trainable local measurement angles as the entangled policy;
- a Bell-entangled quantum policy.

Stage 17 gave every method the same four learning-rate opportunities and two development seeds, then used four separate pilot seeds. It froze method-specific selected rates, 500 updates, final-checkpoint selection, 16 fresh paired confirmatory seeds, 10,000 bootstrap resamples, 8,192 finite-shot decisions per seed, and three noise probabilities.

A positive claim required a paired 95% interval above zero, a one-sided 95% lower bound for the entangled mean above 0.75, and an effect over 0.75 of at least 0.05. No seed could be replaced.

## Confirmatory findings

| Method | Mean exact reward | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Classical local | 0.749925 | 0.749916 | 0.749932 |
| Classical shared randomness | 0.749970 | 0.749970 | 0.749971 |
| Quantum separable | 0.749999 | 0.749996 | 0.750000 |
| Quantum entangled | **0.853553** | 0.853553 | 0.853553 |

The primary comparator was chosen conservatively per seed as the better of classical shared randomness and the separable quantum policy. The entangled-minus-control estimate was 0.10355394, with paired 95% interval [0.10355339, 0.10355476]. The entangled mean's one-sided 95% lower bound was 0.85355339. All three positive gates passed.

The extremely narrow seed interval reflects the small exactly optimized contextual bandit and convergence of every entangled run to the same analytic optimum. It should not be read as uncertainty about larger QMARL systems.

Fresh PennyLane finite-shot evaluation gave 111,745 wins in 131,072 entangled-policy decisions, rate 0.8525467 and Wilson 95% interval [0.8506169, 0.8544558]. Registered exact local-depolarizing sensitivity was 0.8395527 at 2% and 0.8190819 at 5%, both above the classical ceiling.

## Integrity evidence

Stage 16 exhaustively enumerated all 16 deterministic classical policies and certified maximum reward 0.75. Its independent PennyLane circuit matched the analytic Bell correlations, gradients matched finite differences, the separable control reached but did not exceed 0.75, and actor input isolation passed.

Stage 19 validated the 72-file confirmatory bundle, replayed all 64 final checkpoints exactly, reconstructed all 64 raw cell outcomes, reproduced the primary estimate and interval exactly using a separate analysis implementation, and matched the finite-shot totals. The full repository suite passed 124 tests with two PennyLane API deprecation warnings.

The confirmatory run used 32,000 optimizer updates, 524,288 sampled decisions across all four methods, 64,128 circuit or circuit-equivalent evaluations, and 262,144 simulated quantum measurement shots. It used 14.656 experiment wall-seconds and 0.00407 CPU-core-hours.

## Contribution and publication position

The strongest paper contribution is the combined boundary result:

1. two prospective active-SLAM studies show that the tested centralized entangled critic does not reliably improve coverage;
2. a certified decentralized nonlocal game shows a reproducible learned advantage when entanglement expands the actors' feasible correlation set;
3. matched separable quantum and exhaustive shared-randomness controls attribute the positive result specifically to entanglement under the information constraint;
4. finite-shot and registered noise results quantify how the advantage degrades;
5. the complete staged archive demonstrates how to distinguish quantum-model performance from a genuine resource advantage.

This is suitable as a focused QMARL methods, benchmark, or negative-plus-positive mechanism paper. CHSH itself is established, so novelty should be claimed for the controlled QMARL methodology, learned-policy evidence, contrast with critic nulls, and reproducible resource/accounting framework. A stronger general MARL submission would still benefit from a sequential nonlocal Dec-POMDP and physical-device validation, both using new prospective protocols.

## Claim limits

The evidence supports a learned quantum advantage for two decentralized agents in the uniform CHSH contextual bandit with no communication after private inputs. It does not establish computational speedup, physical hardware advantage, advantage in active SLAM, superiority when classical communication is free, or general advantage in sequential MARL. Training used exact expected returns; finite shots were used for frozen-policy evaluation. Noise was simulated with a registered local depolarizing model.

The earlier active-SLAM nulls remain unchanged and should appear in the paper rather than being hidden. Together, the results identify a narrower and more defensible opportunity: quantum correlations can improve decentralized action coordination when the task and information structure make those correlations operationally relevant.
