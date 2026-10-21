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
