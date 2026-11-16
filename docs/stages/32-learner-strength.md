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

Run `32-learner-strength_attempt-01` is registered as planned.
