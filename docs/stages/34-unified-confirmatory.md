# Stage 34 — Unified confirmatory evaluation (test split opened once)

**Why.** In Stages 28 and 30, the learners were evaluated on the sealed v1 test catalogs first, and the strong baselines (bid-price, CE-DP) were registered and evaluated on the same catalogs afterwards. Two consequences followed:
- the paper had to disclose that the comparators came after the test split was opened;
- the v1 test split also leaks dynamics into two catalogs (the Stage-29 audit finding).

This stage removes both. Every method is fixed before a fresh, hash-locked test split is opened, and that split is opened exactly once.

**Test data.** The split is `configs/data/qip_v2/split-lock.json`. Its hash is recorded in the config. The split has 16 request types and four catalogs:
- a 4-agent complete graph;
- a 5-agent ring;
- an 8-agent sparse graph;
- a 6-agent path with H = 24.

Its seeds are collision-free, and its generator rejects any reuse of v1 dynamics or request contents. No script has read its contents before this run. The run reads the split only after `tuning_selection.json` and `training_complete.json` have been written.

**Methods, all fixed before the test split is read:**
- **Greedy.**
- **Bid-price rule**, tuned on the four validation catalogs (500 sequences each) over the 35-point grid, with the development catalogs as reference. Its test reference pools all 12 training catalogs, as in Stage 30.
- **CE-DP with no data**: the default guess, with ρ the mean self-loop probability of the 12 training catalogs.
- **Learners**: 12 fresh PPO models (calibrated value baseline, width 64, learning rate 3e-3, 8,000 episodes) and 12 fresh DQN models (width 64, learning rate 3e-3, 6,000 episodes). These are the Stage-32 validation-selected configurations. All are trained on the 12 training catalogs.
- **References**: online DP (true P) and the future-clairvoyant reference.

**Registered decisions.** All contrasts use the mean over the four catalogs with equal weight, and a seed-level percentile bootstrap with 20,000 resamples.
- **Primary:** PPO minus bid-price.
  - Bid-price superior if the 95% upper bound is below 0.
  - Learned superior if the 95% lower bound is above 0.
  - Inconclusive otherwise.
- **Secondary:**
  - PPO minus greedy: PPO superior if the 95% lower bound is above 0.
  - DQN minus bid-price: same three-way rule as the primary.
  - PPO–DQN equivalence: the 90% interval lies within ±5% of the online-DP mean.
- **Descriptive:** each method's fraction of online DP, CE-DP, the clairvoyant reference, and per-catalog means.

**Budget.** At most 6 wall-hours on 4 CPU workers. No GPU and no shots.

Run `34-unified-confirmatory_attempt-01` is registered as planned.
