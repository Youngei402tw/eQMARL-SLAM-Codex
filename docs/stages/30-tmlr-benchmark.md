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

Run `30-tmlr-baselines_attempt-01` is registered as planned.
