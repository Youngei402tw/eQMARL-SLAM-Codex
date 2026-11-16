# Stage 32 — Learner-strength check

**Why.** The most likely reviewer objection to the TMLR paper is that the learners were under-tuned:
- the best configuration sat at the edge of the grid;
- training was short;
- the PPO value head was mis-scaled;
- 12 PPO seeds collapsed onto 8 distinct outcomes.

This stage tests that objection directly. It uses only BellRoute v1 development and validation data and cannot change any locked decision.

**What changes for the learners.**
- **Grid:** PPO (calibrated value baseline) and DQN each get 6 configurations: width {64, 256} × learning rate {1e-3, 3e-3, 1e-2}. The previous grid stopped at width 64 and learning rate 1e-3 (3e-3 in Stage 31).
- **Training length:** 8,000 PPO episodes and 6,000 DQN episodes, against 2,500 in the locked study.
- **Learning curves:** recorded at 2,000, 4,000, and final episodes.
- **Seeds:** 2 seeds per configuration, then 4 more for each method's selected configuration, giving 6 in total.

**Fairness.**
- The bid-price rule is tuned over the same 35-configuration grid as Stage 30, on the same *selection* sequences (200 per validation catalog) that select the learner configuration.
- The final comparison uses 500 disjoint *comparison* sequences per validation catalog that no selection step has seen.

**Registered decision.** Best method minus rule, bootstrapped over 6 seeds: **gap persists** if the 95% upper bound < 0, **learner superior** if the lower bound > 0, and **inconclusive** otherwise. Learning curves and grid-edge status are reported descriptively.

**Budget.** At most 4.5 wall-hours, 4 CPU workers, no GPU, no shots, and no test data. It is estimated at about 3 hours from a measured 0.08–0.37 s per training episode.

Run `32-learner-strength_attempt-01` was registered before it ran.

## Result: the gap persists

Run `32-learner-strength_attempt-01` ran from source `d66cd394107162f5b30d4d46eb2af5cbd5672e07`. Its status is **completed**: all 36 trials ran and all gates passed. Registered decision: **`gap_persists`**.

**Rule.** The validation-tuned bid-price rule (alpha = 0.75, prior weight = 320) scored:
- 0.3628 on the selection sequences;
- **0.3576 (96.7% of the online DP optimum, 0.3697)** on the fresh comparison sequences.

Greedy reaches 59.3% of the optimum.

**Learner grid (selection mean over 2 seeds).**

| Config | PPO (calibrated) | DQN |
|---|---|---|
| h64-lr1e-3 | 0.3226 | 0.2982 |
| h64-lr3e-3 | **0.3391** | **0.3152** |
| h64-lr1e-2 | 0.3391 | 0.3066 |
| h256-lr1e-3 | 0.3383 | 0.2956 |
| h256-lr3e-3 | 0.3367 | 0.2985 |
| h256-lr1e-2 | 0.3367 | 0.3130 |

**Selected configurations, 6 seeds each, on the fresh comparison sequences.**

| Method | Comparison mean | Of online DP | Minus rule | 95% seed bootstrap | Seeds above rule |
|---|---|---|---|---|---|
| PPO h64-lr3e-3 | 0.3260 | 88.2% | −0.0316 | [−0.0373, −0.0279] | 0 / 6 |
| DQN h64-lr3e-3 | 0.3011 | 81.4% | −0.0565 | [−0.0650, −0.0450] | 0 / 6 |

**Learning curves (selection mean, selected config).**

| Method | 2,000 episodes | 4,000 episodes | Final |
|---|---|---|---|
| PPO | 0.3257 | 0.3352 | 0.3343 (8,000) |
| DQN | 0.3210 | 0.3161 | 0.3043 (6,000) |

**Reading.**
- **The grid is wider than before and the optimum is inside it.** PPO's selected configuration tied with the learning-rate-1e-2 edge (0.3391); ties resolve to the first-listed configuration. The wider network gives no gain.
- **Training longer does not help.**
  - PPO plateaus by 4,000 episodes.
  - DQN gets *worse* with longer training.
- **PPO ends up on one policy.** 8 of the 16 PPO trials reach exactly the same selection score (0.3391) and the same comparison score (0.3302). PPO is converging to a single deterministic allocation policy that sits about 7 points below the rule.
- **The best learner still loses.** The best single learner run scores 0.3302 on comparison, against the rule's 0.3576.
- **Context.** The Stage-26 validation scores (0.30–0.32, on different sequences) are similar. Stronger tuning moved PPO up by a few points at most.

**Conclusion.** The bid-price advantage on BellRoute v1 is not a tuning artifact within this budget (2.4–3.2 times longer training, a 3-times-wider learning-rate range, 4-times-wider networks, a calibrated value baseline, and 6 seeds).

**Resources.**
- 12,073 wall-seconds (3.35 h, within the 4.5 h cap) and 11.86 CPU core-hours.
- 4,586,200 decisions and 1,134,992 updates.
- No GPU, no shots, and no test data.
