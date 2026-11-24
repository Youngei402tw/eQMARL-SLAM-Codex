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

Run `34-unified-confirmatory_attempt-01` was registered before it ran.

## Result

Run `34-unified-confirmatory_attempt-01` ran from source `fd6744e138fe8cf97fc843b4252f5bb6b8bc753a`. Its status is **completed** and all gates passed: all 24 models trained and evaluated, the test certificates are within tolerance (max |SDP gap| 3.4e-10), `tuning_selection.json` and `training_complete.json` were written before the test split was read, and every value is finite. It took 1.80 wall-hours, 3,810,000 decisions and 851,244 updates. The test split was opened once, by this run.

**Bid-price selection (validation only):** α = 0.75, ω = 320, validation mean 0.3584. CE-DP ρ = 0.3463.

**Registered decisions.**
- **Primary: bid-price superior.** PPO − bid-price = -0.047 [-0.053, -0.043]; 0 of 12 PPO seeds above bid-price.
- PPO superior to greedy: **yes**, +0.441 [+0.436, +0.445].
- DQN vs bid-price: **bid-price superior**, -0.180 [-0.204, -0.157].
- PPO–DQN equivalence: **not equivalent**. PPO − DQN = +0.133, 90% CI [+0.113, +0.154], margin ±0.044. PPO is better.

**Descriptive.** PPO − CE-DP (no data) = -0.059 [-0.065, -0.055].

Fraction of online DP (catalog-averaged): greedy 42.6%, DQN 77.9%, PPO 93.1%, bid-price 98.5%, CE-DP (no data) 99.9%, online DP 100%, future-clairvoyant 104.7%.

Per catalog (% of online DP):

| Catalog | Greedy | Bid-price | CE-DP | PPO | DQN | Clairvoyant |
|---|---|---|---|---|---|---|
| qip2-test-a4-complete | 52.3 | 98.3 | 99.9 | 86.8 | 93.9 | 103.8 |
| qip2-test-a5-ring | 41.2 | 98.1 | 99.8 | 95.2 | 81.4 | 105.5 |
| qip2-test-a8-sparse | 39.5 | 98.7 | 99.8 | 94.0 | 70.3 | 105.3 |
| qip2-test-a6-path-h24 | 41.8 | 98.6 | 100.1 | 94.5 | 76.5 | 103.8 |

**Observations (descriptive, not registered).**
- 10 of 12 PPO seeds reached exactly the same test score (0.8171); the calibrated PPO configuration converges to nearly the same policy across seeds, so the seed bootstrap interval for PPO is narrow.
- DQN beats PPO on the 4-agent complete graph; PPO is ahead on the other three catalogs.

**Interpretation.** Under a single opening of a fresh test split, with every method fixed beforehand, the conclusion of the earlier stages holds: the validation-tuned bid-price rule beats the best learner, and CE-DP with no data comes within 0.1% of online DP. The learners clearly beat greedy, so they learn pacing, but not as well as a simple threshold.
