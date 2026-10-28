# Stage 14 — Mechanism-study reproduction and paper release

Status: audit passed from exact Stage 13 completion commit `d177c35cbd6b44cac12fa5f71eae82ce5e3b9d79`.

The release audit validates all 120 confirmatory bundles, loads every final checkpoint, replays one validation scenario for each of the 15 method-condition cells, independently recomputes the registered paired interaction and bootstrap interval, reconciles resources and failures, and emits paper-ready tables plus a claim/limitation matrix.

No training or additional test evaluation is authorized in this stage. The audit is a bounded reproduction and release check.

The remotely published audit source was `bfefca24b43b619435b963931eb8b27af99f676c`. Its single run passed every gate:

- all 120 confirmatory bundles validated against inventories and the locked source;
- all 120 final checkpoints loaded and contained finite tensors;
- one validation scenario for every method-condition cell replayed exactly (15/15, maximum error 0.0);
- the primary +0.007083 estimate and [-0.155974, 0.180293] interval reproduced exactly;
- all confirmatory runs were completed and the failure ledger was empty;
- the 534,606 analytic circuit evaluations and zero physical shots reconciled.

The audit itself used 183 primitives, 224 agent decisions, zero optimizer updates, zero circuit evaluations, zero shots, 11.432 wall-seconds, and 0.00317 CPU-core-hours. It did not train or access the held-out test manifest.

The paper report is `docs/MECHANISM_STUDY_REPORT.md`. Machine-readable paper tables, reproduced inference, and the claim/limitation matrix are under `analysis/14/`.

Stage gate: all clean-reproduction, checkpoint, analysis, failure/resource-ledger, and paper-artifact requirements passed. Fresh remote publication verification remains pending.
