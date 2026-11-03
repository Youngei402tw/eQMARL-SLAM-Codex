# Stage 27 — robustness, scaling, and confirmatory freeze

## Preflight

**Question.** Does resource-aware allocation remain useful across non-final changes in graph size, topology, horizon, budget, entanglement visibility, and heralded generation success?

**Smallest falsifier.** Train fresh seeds for the four frozen Stage 26 configurations on development catalogs, then evaluate the unchanged policies on all validation catalogs. A collapse to greedy behavior under modest visibility `0.8` or heralded success `0.75`, invalid actions, or sensitivity confined to one topology falsifies a broad robustness explanation. Every null condition remains in the archive.

**Deliverables.** Twelve fresh checkpoints; paired condition/catalog summaries for DQN, Double DQN, PPO, and recurrent PPO; ideal greedy and model-aware references under every condition; parameter, training, and inference costs; positive-gain prevalence; and a hash-locked Stage 28 configuration.

**Budget.** Three seeds per method, 1,800 episodes per seed, 200 paired validation sequences per catalog and condition, at most 1.5 million decisions, and three active wall-hours. One job may use at most four CPU threads.

**Gate.** Complete every registered condition across three, four, and six agents and path, ring, star, and sparse-random graphs; retain all outcomes; remain within the cap; and leave the eight-agent/test request family unopened. Scientific robustness is reported rather than required for stage completion.
