# Stage 28 — locked robust entanglement-routing evaluation

## Preflight

**Question.** On fresh request games and public dynamics, including unseen six-agent ring and eight-agent sparse graphs, does validation-selected PPO improve over greedy allocation and how does it compare with the strongest conventional DQN comparator?

**Falsifiers.** PPO superiority over DQN requires a seed-paired 95% bootstrap lower bound above zero. PPO and DQN equivalence requires the paired 90% interval to lie wholly within ±5% of the online-reference surplus. PPO improvement over greedy requires a seed-level 95% lower bound above zero. These outcomes are independent; every null or negative result will be retained.

**Locked design.** Train twelve fresh PPO and twelve fresh DQN policies for 2,500 episodes on combined development/validation catalogs. Evaluate final checkpoints on 500 paired sequences for each of four sealed catalogs. Classical exact, random, static, greedy, online model-aware, future-clairvoyant, and message oracles are reported with their information differences. Four frozen noisy-resource conditions use 100 paired sequences per catalog for PPO to remain within the 2.5-million-decision cap; this secondary count is fixed before opening test values.

**Gate.** Complete all 24 models, all primary and secondary evaluations, 20,000 bootstrap resamples, Holm-adjusted catalog contrasts, and safety checks within four active wall-hours and 2.5 million decisions. A scientific null result passes the execution stage when complete and correctly reported.
