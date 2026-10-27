# Stage 12 — Equal-budget mechanism development and freeze

Status: running from exact Stage 11 completion commit `7442b2354fc5f630c3c0753697f60078682f7b10`.

The first registered phase gives each of five critics the same two learning rates, two seeds, 5,000 training primitives, and 30 greedy validation scenarios in the low-complementarity condition. Mean validation return across the two seeds selects one rate per method, with the lower rate winning exact ties. Every result, including failures, remains in the archive.

The 20 tuning runs all completed and passed their numerical gates. The selected rates are `0.001` for `central_classical`, `0.003` for `bilinear`, `0.003` for `flexible_interaction`, `0.001` for `quantum_noent`, and `0.003` for `eqmarl`. The complete two-seed table is in `analysis/12/tuning-selection.json`. The flexible critic's width was selected structurally from widths 8, 16, and 32 before pilot outcomes: width 16 has 250 parameters and is closest to the 249-parameter centralized control.

After rate selection, a fixed pilot matrix will run every method in additive, low, and high complementarity with three paired seeds and 10,000 primitives. Pilot outcomes may describe variance and feasibility but cannot change the primary contrast, methods, conditions, confirmatory seeds, or checkpoint rule. The mechanism test catalog is not available to either phase.

All 45 pilots completed and passed. Mean validation physical coverage for eQMARL minus matched no-entanglement was -0.13827 in additive, -0.15432 in low, and -0.12562 in high complementarity. The descriptive pilot high-minus-additive interaction was +0.01265. This is below the registered 0.02 minimum effect of scientific interest and is not confirmatory evidence; the three-seed pilot was retained only for feasibility and variance description. It did not change any method, condition, seed, budget, contrast, or checkpoint rule.

Across tuning and pilots, all 65 planned outcomes completed. They consumed 591,222 primitive steps, 447,388 agent decisions, 23,692 optimizer updates, 250,430 analytic circuit evaluations, zero shots, 10,820.610 experiment wall-seconds, and 3.005 CPU-core-hours. The complete pilot table and seed rows are in `analysis/12/pilot-phase-diagram.json`.

The Stage 13 lock fixes five methods, three conditions, eight seeds, 10,000 training primitives, 30 validation episodes, 100 held-out test episodes, final checkpoints, the paired difference-in-differences estimand, a 10,000-resample paired-seed bootstrap, bootstrap seed 14001, and the 0.02 minimum effect of scientific interest. Test scenarios are deferred to runtime behind their registered SHA-256 and cannot enter tuning or pilot code.

Stage gate: numerical, information, retention, equal-budget, and feasibility requirements passed. Remote result verification and the Stage 13 source/config publication remain before any mechanism-test access.
