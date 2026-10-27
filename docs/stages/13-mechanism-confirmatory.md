# Stage 13 — Locked complementarity interaction phase diagram

Status: preparing the locked source from exact Stage 12 completion commit `d733d2b99296e27c7678c6cbbbfd8f03848b1d34`.

The confirmatory matrix is fixed at five critics, additive/low/high complementarity, and eight paired trained seeds: 120 models. Every model receives 10,000 training primitives, 30 greedy validation episodes, and 100 paired held-out test episodes. The final checkpoint is evaluated actor-only, with no replacement, exclusion, retuning, or outcome-based extension.

The primary estimand is the seed-paired difference-in-differences in physical coverage: `[eqmarl - quantum_noent]_high - [eqmarl - quantum_noent]_additive`. Its two-sided 95% interval uses 10,000 paired-seed bootstrap resamples with seed 14001. The registered minimum effect of scientific interest is 0.02; this is not an equivalence test.

The held-out scenario manifest will be loaded only by remotely verified Stage 13 execution. Runtime rejects a path, hash, split, or count mismatch before test evaluation.
