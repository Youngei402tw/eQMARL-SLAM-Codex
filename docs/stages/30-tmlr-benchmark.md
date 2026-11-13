# Stage 30 — Machine-learning venue revision

## Preflight

**Question.** Does the learned allocators' advantage over greedy allocation survive a stronger non-learned heuristic, and how much logged public data does a model-based planner need to match the zero-shot learned allocators? These are the two baselines that a machine-learning reviewer would ask for first.

**Target venue.** *Transactions on Machine Learning Research* (TMLR). TMLR judges submissions on whether the claims are correct and supported by evidence and whether some part of the community would find them interesting, rather than on novelty. It reviews double-blind, so the manuscript is anonymized.

**Smallest falsifying experiment.** If the 95% seed-bootstrap upper bound of PPO minus the tuned bid-price heuristic is below zero on the sealed catalogs, a simple pacing rule beats the learned allocator and the benchmark's main learned claim must be withdrawn.

**Comparators** (`eqmarl_slam/routing_baselines.py`):
- *Bid-price (budget-pacing) threshold.* It serves the best feasible matching whose every edge has gain at least the `1 - alpha*b/(2k)` quantile of a reference distribution of per-slot gains. That reference mixes training-catalog gains (total weight `prior_weight`) with gains observed so far in the episode. Two parameters, a 7 x 5 grid. It never sees the transition matrix.
- *Certainty-equivalent DP.* It estimates the public transition matrix from `K` logged public sequences of the catalog under a Dirichlet prior centred on the development mean self-loop, and then plans exactly. `K` is in {0, 1, 3, 10, 30, 100}, and the prior mass `kappa` (5 values) is selected per `K`. It needs the catalog's public state table and logged data that the learned allocators never receive. It is therefore an informed reference, not a fair competitor.

**Protocol.**
1. Tune both comparators on the four validation catalogs only (500 sequences each). The development catalogs supply the heuristic's reference distribution. Write `tuning_selection.json` before any sealed file is read.
2. Evaluate the selected configurations once on the four sealed catalogs, using the same 500 paired sequences as Stage 28.
3. Compare them with the archived Stage-28 learned rows. No checkpoint is loaded, retrained, or reselected.

The registered decision rule is in `configs/stage30/30-tmlr-baselines_attempt-01.yaml`. Fully fresh catalogs are reported separately because of the Stage-29 overlap finding.

**Post-lock status.** The sealed catalogs were opened in Stage 28. This run adds comparators. It does not alter any Stage-28 decision, and its results are labelled as post-lock in the manuscript.

**Gates.**
- Input hashes are verified.
- Tuning is completed and written before the test catalogs are loaded.
- The evaluator reproduces the archived Stage-28 greedy and online-DP rows to within 1e-9.
- All methods and catalogs are complete.
- The paired learned rows are complete.
- All values are finite.

**Budget.** At most one wall-hour, two CPU threads, 3,000,000 allocator decisions, no training, and no shots. A short benchmark measured under 1 ms per episode.

**Also delivered.**
- `eqmarl_slam/bellroute_env.py`, a Gymnasium environment that reproduces the archived evaluator exactly (tested).
- The anonymized TMLR manuscript under `docs/paper/tmlr/`.

Run `30-tmlr-baselines_attempt-01` was registered as planned. It passed every execution gate, but the final archive validation rejected it because the run-manifest schema's stage enum stopped at `29`. That bundle is retained as `failed_archive_validation`. After the schema fix, run `30-tmlr-baselines_attempt-02`, identical apart from its run ID, is registered as planned.

## Result

Run `30-tmlr-baselines_attempt-02` ran from source `03d4b07edc75862960d77c0ec070c33dfa2daa53`. Its status is **completed** and all six execution gates passed. The evaluator reproduced the archived Stage-28 greedy and online-DP rows with a maximum absolute error of 0.0. The retained attempt-01 bundle produced identical method means.

**Validation-only selection.** The selected bid-price configuration was alpha = 0.75 and prior_weight = 80, with a validation mean of 0.3534. The best Stage-26 learners scored 0.3235–0.3247 on the same validation catalogs (under different sequences). For certainty-equivalent DP, the development self-loop was rho = 0.3470, and the selected kappa for each K was {"0": null, "1": 8, "3": 128, "10": 8, "30": 0.5, "100": 0.5}.

**Registered primary decision: `bid_price_superior`.** PPO minus bid-price was -0.0890, with 95% CI [-0.0964, -0.0821]. The heuristic was ahead of all 12 PPO seeds. DQN minus bid-price was -0.0772, with 95% CI [-0.0864, -0.0675]. On the two fully fresh catalogs, PPO minus bid-price was -0.1035, with 95% CI [-0.1124, -0.0950].

The registered falsifier therefore fired: a two-parameter pacing rule tuned on validation data dominates both learned allocators. The Stage-28 result that the learners beat greedy still stands, but the claim that learned allocators are a strong solution to this benchmark is withdrawn.

**Certainty-equivalent DP** matched online DP even with K = 0 logged sequences (registered break-even K = 0). With the catalog's public state table and only a generic self-loop prior, planning loses almost nothing relative to knowing the true transition matrix. Most of the learners' shortfall is therefore budget pacing against the future gain distribution, not dynamics prediction.

| Method | Mean (all) | Of online DP | Mean (fresh) | Of online DP (fresh) |
|---|---|---|---|---|
| greedy_recomputed | 0.4183 | 51.9% | 0.5212 | 53.7% |
| online_dp_recomputed | 0.8060 | 100.0% | 0.9714 | 100.0% |
| bid_price | 0.7836 | 97.2% | 0.9386 | 96.6% |
| ce_dp_k0 | 0.8052 | 99.9% | 0.9704 | 99.9% |
| ce_dp_k1 | 0.8045 | 99.8% | 0.9687 | 99.7% |
| ce_dp_k3 | 0.8054 | 99.9% | 0.9705 | 99.9% |
| ce_dp_k10 | 0.8030 | 99.6% | 0.9658 | 99.4% |
| ce_dp_k30 | 0.8043 | 99.8% | 0.9703 | 99.9% |
| ce_dp_k100 | 0.8057 | 100.0% | 0.9711 | 100.0% |
| ppo | 0.6946 | 86.2% | 0.8352 | 86.0% |
| dqn | 0.7065 | 87.7% | 0.8351 | 86.0% |

**Resources.** 77.5 wall-seconds, 0.0210 CPU core-hours, 1,736,000 allocator decisions, no training, and no shots. Attempt-01 used about the same.

## Manuscript

The anonymized TMLR manuscript is `docs/paper/tmlr/BELLROUTE_TMLR.{tex,pdf}`. It has 14 pages and builds with no LaTeX errors, overfull boxes, or undefined references. It is reframed around the registered outcome:
- The learned allocators beat greedy, as registered in Stage 28.
- A validation-tuned bid-price rule beats every learned seed, as registered in Stage 30.
- Certainty-equivalent DP with no logged dynamics is near-optimal.
- The split leak found by the audit is disclosed, and every result is also reported on the fully fresh catalogs.

All 195 result macros the paper uses come from `docs/paper/tmlr/generated/numbers30.tex` (this stage) and `docs/paper/generated/numbers.tex` (the Stage-29 audit), and all appear verbatim in the PDF text. Data figures are built from `docs/paper/tmlr/figure_data/*.csv`, which are hashed in `SHA256SUMS`. Before submission, the author must:
- prepare anonymized supplementary material;
- avoid dual submission with the QIP draft.
See `docs/paper/tmlr/SUBMISSION_CHECKLIST.md`.
