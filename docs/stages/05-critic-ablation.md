# Stage 05 — Matched critic attribution

Status: running. Dependency completion: `2f40002ca55f05ff620d25872971241d8d730223`.
The branch starts from that exact verified Stage04 state.

Question: does critic choice alter value fitting or bounded recurrent learning
when actor, task, information, compression dimensions and rollouts are common?
The smallest falsifier is a disconnected encoder/circuit gradient or a no-input-
entanglement control with any other changed parameter/configuration.

The initial matrix includes shared split classical, compressed centralized
classical, bilinear, matched split quantum without/with Bell input, and qfCTDE
as an additional global topology. Every critic has a trainable shared27to2 local
encoder; all corrected heads are affine with return scale10. The actor/GRU,
taskv2, candidates/planner, macro duration and PPO semantics remain fixed.
Actor weights will train independently for RL; only the registered fixed-corpus
diagnostic uses frozen Stage04 actors. No faithful/A2C result enters this ranking.

First collect a real return dataset from all3 archived pilot actors, equally
represented on fixed training/validation maps. Fit each critic for32 full-batch
Adam updates at0.01, record signs/variance/predictions/gradients and validation
error, and test fresh models on a separate signed two-point capacity fixture
for128 updates. Capacity MSE gate0.0025, finite gradients and encoder connection
are required. Real-data underfitting is a valid negative result. Identical state
dictionaries verify the matched input-entanglement intervention.

Diagnostic cap1h and stage cap24 cumulative active hours; one CPU thread/job,
no new compute. Count actual device executions across forward and backward.
Equal-budget optimizer trials and bounded pilots are registered only after
these correctness gates and throughput are available. Final test data remain
unused. All configurations, raw outcomes and checkpoints are permanent artifacts.

All13 fitting gates passed (87 simulated primitives,960 regression updates,3045 device executions,43.57sec). Real targets were exclusively positive; signed fixture is separate. See immutable fitting bundle.

The registered tuning matrix uses2 rates(0.001,0.003),2 seeds and1000 training primitives for every method, then3 pilots of2000 steps with30 greedy validation episodes. Select by two-seed mean validation return, lower-rate exact tie. This common short budget measures bounded learning, not convergence. Final test remains unopened.

Mechanical archive recovery: first4 bilinear runs finished numerically but ID decimal points violated the old schema. No training was interrupted. Original bundles/logs retained; an explicit schema amendment admits safe decimal labels and validates identity before execution. Future runs stop on post-run archive errors. Model/task/training code unchanged.

Tuning complete. Rate decisions and all per-seed outcomes are in `configs/stage05/learning-rate-selection.json`; the registered selection rule is applied without modification. Pilot configs are published before execution.
