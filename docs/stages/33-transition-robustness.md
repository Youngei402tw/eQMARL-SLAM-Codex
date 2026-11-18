# Stage 33 — Robustness to the transition matrix

**Why.** All BellRoute v1 catalogs come from one generator. Its dynamics mix fast: the probability of staying in the same state is 0.31–0.37 in every training catalog.

Stage 30 drew two conclusions from those catalogs:
- **(H1)** a budget-pacing (bid-price) rule captures most of the optimum;
- **(H2)** the dynamics barely matter. CE-DP, which uses only the public state table and a generic default guess of the transition matrix, reached 99.9% of online DP.

The user asked whether these conclusions survive other transition matrices.

**Exploratory disclosure.** Before registration, an unregistered sweep of six of the variants below was run on the 8 development catalogs (150 tuning and 400 evaluation sequences each). Results:
- CE-DP with the default guess kept 98.8–99.6% under every variant except a strongly structured one ("successor 0.8"). There it dropped to 96.1% overall and 81.8% on the worst catalog.
- The bid-price rule kept 93.9–96.9% throughout.

The registered thresholds below were written with that result known. H2 is therefore *expected* to fail for structured dynamics. The script and its output are not part of the archive.

**Design.** Each of the 4 validation catalogs is rebuilt under 7 registered variants. Each variant keeps the catalog's state table, horizon, and budget, and changes only the transition matrix P.

| Variant | Transition matrix |
|---|---|
| as generated | unchanged |
| stay 0.05 / 0.7 / 0.9 | stay probability set to s; off-diagonal mass keeps the original relative proportions |
| successor 0.5 / 0.8 | mix of a fixed derangement (each state has one likely successor, seeded by catalog id) with weight w, and the original P |
| rare jackpot | the state with the highest one-round value is entered about 12.5× less often and persists with probability 0.5 once entered |

**Methods scored**, on 500 evaluation sequences per catalog:
- greedy;
- bid-price frozen at the Stage-30 selection (α = 0.75, ω = 80);
- bid-price retuned per variant over the 35-point grid, on 200 disjoint tuning sequences;
- CE-DP with the default guess (ρ = 0.347, the development mean self-loop; no data);
- CE-DP with the stay probability matched to the variant;
- CE-DP with 10 and 100 logged episodes of the variant (κ frozen from Stage 30);
- online DP, the reference.

No learner is trained and no test data are read.

**Registered decisions.** All use the overall fraction of online DP.
- **H1:** retuned bid-price ≥ 90% under every variant.
- **H2:** CE-DP with the default guess ≥ 98% under every variant. Failing variants are listed.
- **H3:** wherever H2 fails, CE-DP with 100 logged episodes ≥ 98%.

Sequence-bootstrap 95% intervals and worst-catalog fractions are descriptive.

**Pre-run smoke disclosure.** After the config was written, a 50-sequence timing smoke ran on the validation catalogs. Its numbers resembled the exploratory sweep. The only change made afterwards was to the wall-time cap (5,400 → 1,800 s). No variant, method, threshold, or seed was changed.

**Budget.** At most 0.5 wall-hours on 4 CPU workers. No GPU, no shots, no training, and no test data.

Run `33-transition-robustness_attempt-01` is registered as planned.
