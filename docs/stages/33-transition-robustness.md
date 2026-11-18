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

Run `33-transition-robustness_attempt-01` was registered before it ran.

## Result

Run `33-transition-robustness_attempt-01` ran from source `07ecbbf5d265a8b37cf6106c45c70ee6c3ef68a2`. Its status is **completed**: all 7 variants ran and all gates passed. It took 58 wall-seconds and 3,542,000 decisions.

> **Note.** The first launch, from source `6910eb5`, was refused before any work started. The runner's dependency check rejected Stage 32's registry status "completed". That status was corrected to "passed" in `07ecbbf` (no result changed), and the run was relaunched. No partial bundle was produced.

**Registered decisions.**
- **H1 (pacing generalizes): supported.** Retuned bid-price scored 94.9–97.8% of online DP under every variant, against a threshold of 90%.
- **H2 (dynamics irrelevant everywhere): not supported, as anticipated.** CE-DP with the default guess fell below 98% only under **successor 0.8**, at 96.5%.
- **H3 (logged data recovers): supported.** Under successor 0.8, CE-DP with 100 logged episodes reached 99.9%; 10 episodes already gave 98.9%.

**Fraction of online DP** (overall over the 4 validation catalogs; 95% sequence-bootstrap interval; worst catalog in parentheses):

| Variant | Mean stay | Greedy | Bid-price, frozen | Bid-price, retuned | CE-DP default | CE-DP, matched stay | CE-DP, K=10 | CE-DP, K=100 |
|---|---|---|---|---|---|---|---|---|
| as generated | 0.34 | 58.5 | 96.8 | 96.8 [96.4, 97.1] (90.1) | 99.8 [99.6, 100.0] (99.6) | 99.8 | 99.7 | 100.0 |
| stay 0.05 | 0.05 | 55.9 | 97.2 | 97.6 [97.3, 97.9] (95.7) | 99.6 [99.4, 99.8] (99.4) | 99.8 | 99.6 | 99.9 |
| stay 0.7 | 0.70 | 64.2 | 96.0 | 96.4 [96.0, 96.9] (94.8) | 99.4 [99.1, 99.6] (99.1) | 99.4 | 99.4 | 99.8 |
| stay 0.9 | 0.90 | 71.5 | 95.8 | 96.4 [95.8, 96.9] (95.0) | 98.6 [98.2, 98.9] (98.2) | 99.6 | 99.2 | 99.6 |
| successor 0.5 | 0.17 | 58.7 | 95.8 | 96.4 [96.0, 96.9] (92.9) | 98.9 [98.6, 99.2] (98.6) | 98.9 | 99.7 | 100.0 |
| successor 0.8 | 0.07 | 59.3 | 94.0 | 94.9 [94.5, 95.4] (89.2) | **96.5** [96.1, 96.9] (95.9) | 96.7 | 98.9 | 99.9 |
| rare jackpot | 0.37 | 63.0 | 95.7 | 97.8 [97.5, 98.1] (94.2) | 99.3 [99.0, 99.5] (98.9) | 99.3 | 99.5 | 99.9 |

**Reading.**
- **The stay probability barely matters.** CE-DP's default guess stays at 98.6% or higher from stay 0.05 to 0.9, although the guess (0.347) is far off at both ends. At stay 0.9, matching the stay probability recovers 99.6%.
- **What the default guess misses is predictable structure** (a likely successor for each state). Under successor 0.8 it loses 3.5 points. A few logged episodes recover nearly all of it.
- **Pacing is robust.** The bid-price rule stays at 94–98% across all variants, and greedy at 56–72%. The Stage-30 conclusion that a pacing rule captures most of the optimum holds across these dynamics.
- **The frozen Stage-30 bid-price parameters transfer well**, within 2.1 points of the retuned rule.

**Interpretation for the paper.**
- The claim that "dynamics are nearly irrelevant" holds for fast-mixing or merely persistent chains.
- It does not hold for chains with strong successor structure. There, knowing P is worth a few points, which a planner can recover from little data.
- The main conclusion, that strong pacing rules leave little headroom for learning, is unchanged.
- Caveats:
  - only validation catalogs (horizon 8–24, budget 2–6) were tested;
  - no learner was trained on these variants.
