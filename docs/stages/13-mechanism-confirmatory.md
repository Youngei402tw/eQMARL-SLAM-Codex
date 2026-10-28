# Stage 13 — Locked complementarity interaction phase diagram

Status: completed null result from exact Stage 12 completion commit `d733d2b99296e27c7678c6cbbbfd8f03848b1d34` and locked source `e64f53330f0f108dd26f94e0172d3f629c2639b6`.

The confirmatory matrix is fixed at five critics, additive/low/high complementarity, and eight paired trained seeds: 120 models. Every model receives 10,000 training primitives, 30 greedy validation episodes, and 100 paired held-out test episodes. The final checkpoint is evaluated actor-only, with no replacement, exclusion, retuning, or outcome-based extension.

The primary estimand is the seed-paired difference-in-differences in physical coverage: `[eqmarl - quantum_noent]_high - [eqmarl - quantum_noent]_additive`. Its two-sided 95% interval uses 10,000 paired-seed bootstrap resamples with seed 14001. The registered minimum effect of scientific interest is 0.02; this is not an equivalence test.

The held-out scenario manifest will be loaded only by remotely verified Stage 13 execution. Runtime rejects a path, hash, split, or count mismatch before test evaluation.

## Confirmatory result

All 120 registered models completed, and every final checkpoint passed frozen actor-only and complete 100-episode test gates. No run was replaced, excluded, retuned, or extended. The primary paired-seed interaction was **+0.00708** physical coverage with a two-sided 95% paired bootstrap interval **[-0.15597, +0.18029]**. The point estimate is below the registered 0.02 minimum effect of scientific interest and the interval spans zero. This is a completed null result, not an equivalence finding.

Mean held-out results across eight trained seeds and 100 test episodes per seed were:

| Condition | Method | Physical coverage | Success | Return |
|---|---:|---:|---:|---:|
| additive | central classical | 0.86455 | 0.87500 | 5.88040 |
| additive | bilinear | 0.81813 | 0.62375 | 5.88415 |
| additive | flexible interaction | 0.94844 | 0.99875 | 6.63158 |
| additive | quantum no-entanglement | 0.88851 | 0.86000 | 5.46816 |
| additive | eQMARL | 0.81257 | 0.59750 | 5.87679 |
| low | central classical | 0.85722 | 0.87375 | 5.95743 |
| low | bilinear | 0.85531 | 0.76250 | 5.88223 |
| low | flexible interaction | 0.83469 | 0.62375 | 6.13064 |
| low | quantum no-entanglement | 0.80559 | 0.59125 | 4.95122 |
| low | eQMARL | 0.78979 | 0.63250 | 5.27545 |
| high | central classical | 0.77878 | 0.43500 | 6.01832 |
| high | bilinear | 0.81677 | 0.42500 | 6.44060 |
| high | flexible interaction | 0.75642 | 0.35500 | 5.98260 |
| high | quantum no-entanglement | 0.79653 | 0.32625 | 5.54350 |
| high | eQMARL | 0.72767 | 0.31625 | 5.54406 |

In the high condition, descriptive eQMARL coverage differences were -0.05111 versus central classical (95% unadjusted interval [-0.23185, 0.12153]), -0.08910 versus bilinear ([-0.21861, 0.03646]), and -0.02875 versus flexible interaction ([-0.10698, 0.05826]). These intervals are descriptive and unadjusted.

The fixed analysis initially stopped before inference because it requested the secondary key `travel_distance`; raw actor-only records use `travel_distance_m`. The reporting code changed only that field name. The primary metric, estimand, bootstrap seed/resamples, raw results, and every run remained unchanged.

The matrix consumed 1,501,711 primitives, 1,242,090 agent decisions, 54,740 optimizer updates, 534,606 analytic circuit evaluations, zero shots, 27,929.729 experiment wall-seconds, and 7.757 CPU-core-hours. A reclaimed session stopped between runs after 114 bundles; the remaining six previously unstarted configs resumed from the same locked source. There was no partial run or retry.

Fresh independent retrieval of results commit `33d108b578e6a87c5d890280039932a30f19f7e6` validated all 120 bundles and every resolved config, raw metric table, diagnostic, checkpoint, manifest, and inventory hash. The receipt is `docs/stages/13-publication.json`.

Stage gate: completed null result. The numerical, actor-only evaluation, explicit-outcome, raw-traceability, budget, and fresh-publication requirements passed.
