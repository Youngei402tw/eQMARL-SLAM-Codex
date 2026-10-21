# Stage 05 — Matched critic ablations

Status: passed. Dependency: `2f40002ca55f05ff620d25872971241d8d730223`.

Question: does critic choice change fitting and short-budget recurrent learning when actor, task, information, local compression and rollout semantics are common? Disconnected gradients or a non-matched Bell intervention would block the comparison. These correctness falsifiers passed.

All six methods use trainable shared27to2 critic compression and an affine value head with return scale10. The shared16-unit recurrent actor is trained independently per method. Task `belief-controller-v2`, estimator, planner, masks, duration-aware PPO/GAE and greedy validation remain fixed. qfCTDE is an additional topology, not the no-entanglement control. eQMARL/noent initialize every parameter identically and differ only in input Bell preparation; both selected rate0.001.

The fitting diagnostic used all three Stage04 actors with equal episode allocation; different episode lengths yielded unequal transition counts. Its31 return rows were all positive. Every method passed meaningful gradient and separate signed two-point capacity checks. The signed fixture establishes capacity, not empirical negative Monte Carlo returns. PPO targets subsequently spanned negative and positive values. Real-corpus underfitting remains reported.

Equal tuning: two predefined rates(0.001,0.003), two seeds per rate,1000 training primitives each and30 greedy validation episodes. The fixed rule maximized two-seed mean validation return, lower-rate exact tie. Rate0.003 selected for centralized classical;0.001 for all others. Each method then used three paired independent actor seeds5401–5403 at2000 training primitives and30 validation episodes per final checkpoint. These are common bounded budgets, below provisional20k pilots; no convergence claim.

Means below average the three trained seeds; episodes are not independent training replications.

| Critic | Coverage | Success | Return | Pilot seconds | Circuit calls |
|---|---:|---:|---:|---:|---:|
| split_classical | 0.918519 | 0.977778 | 5.906243 | 85.39 | 0 |
| central_classical | 0.954630 | 1.000000 | 6.367232 | 77.23 | 0 |
| bilinear | 0.891667 | 0.822222 | 4.982953 | 96.88 | 0 |
| quantum_noent | 0.848457 | 0.655556 | 5.158994 | 207.06 | 7379 |
| eqmarl | 0.925926 | 0.922222 | 6.046653 | 178.26 | 6676 |
| qfctde | 0.848148 | 0.644444 | 5.103331 | 215.66 | 7351 |

Centralized classical had the strongest validation return and90/90 successes, supporting a functioning baseline. eQMARL exceeded noent in this exploratory sample, substantially driven by seed5401 of the control, and remained below centralized classical. Three seeds cannot establish a robust advantage. All greedy stalls, collisions, imperfect maps and seed variation remain in raw records. No faithful/A2C reference is pooled into this corrected leaderboard.

All18 frozen pilot actors replayed the first registered validation map exactly within1e-8 over physical/estimated coverage, map accuracy, pose/trajectory error, travel, collisions, communication, return and duration. This separate diagnostic records action/duration/reward-component counts without optimizer or critic execution. Original pilot training action sequences were not retained; replay counts are labeled separately. Subsequent logging records action/reward counts, target/value moments and gradient groups without changing optimization.

Mechanical recovery: the first four bilinear tuning runs finished numerically but their decimal rate labels violated the old archive-ID regex. Original bundles and error logs are retained. An explicit safe-period schema amendment and early ID check fixed validation; no numerical run was interrupted or retrained. The batch now stops on post-run archive errors. See [recovery record](../../configs/stage05/archive-recovery.json).

Sources: fitting `614879ad91f971e2ab55177cbd83d32f13f75512`; initial tuning `66eac426233beaafb8280a6fe70b16ed734360b0`; validation-only recovery source `00566b9422ee5e580d4690bbe75ec612e05c422a`; all pilots `ebac5755184ce6558fcf35cdbd8d1024d5039882`; frozen replay `6d3b7724e1eb89c81d9541706e72aaed578eebba`. Complete source/config/results identities are in the [run index](../../experiments/05/index.json). All44 bundles and every inventoried fitted/final checkpoint were independently retrieved at results commit `d6c16db627f511ab04a4caca7ea39e7ec32bd216`; see [publication receipt](05-publication.json).

The [pilot analysis](../../analysis/05/pilot-analysis.json) traces every seed to raw hashes. The [rate decision](../../configs/stage05/learning-rate-selection.json) retains every tuning outcome. [Constructor resolution](../../configs/stage05/effective-model-contract.json) supplements immutable early configurations.

Recorded experiment resources:76016 primitives(including validation/diagnostics),56814 agent decisions,3758 updates(including regression),39373 analytic device executions,1619.38sec wall,0.44977 measured CPU core-hours,zero GPU hours. No physical shots sampled. Stage active envelope is recorded in project state and remains below24h; no new compute.

Gates: fitting/gradient/intervention checks passed; all42 task-optimizer runs and18 pilot policies retained; all18 replay checks passed; required artifact availability verified. Local suite106 tests and planning checks passed. Final test remains unaccessed. Proceed to Stage06 component attribution and retain centralized classical as the validation-selected strong task control.
