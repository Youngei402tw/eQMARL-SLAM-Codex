# Stage 09 — Reproducible release and handoff

Status: running. The clean-runtime audit is preregistered; no Stage 09 reportable run has started.

## Preflight

Question: can a fresh environment built only from the committed lockfile verify every frozen Stage 08 checkpoint, replay saved policies, and regenerate the locked analysis without retraining or mutating evidence?

The smallest falsifier is one actor-only replay of the first validation scenario for seed 8801 from each of the three confirmatory methods. Any checkpoint hash/load failure, replay error at or above `1e-8`, or byte difference in regenerated analysis fails the diagnostic. The full audit also validates all 24 checkpoint bundles because missing or corrupted evidence would invalidate the release even if the three replays passed.

Deliverables are the exact clean-runtime inventory, complete resolved configuration and checkpoint-hash registry, raw checkpoint/replay/regeneration records, diagnostic gates, frozen outputs, full archive/resource audit, final report, and continuation guide.

The run uses one CPU thread, at most 108 primitives, at most 3,600 wall-seconds, zero optimizer updates, zero circuit evaluations, and zero shots. It reads validation data for the replay and archived Stage 08 raw data for analysis regeneration. It does not retrain, select a policy, or alter the held-out result.

Exit requires all checkpoint and archive hashes to validate, three replays to match below `1e-8`, locked analysis to regenerate byte-for-byte, every planned run to have a retained outcome, the remote archive to be independently retrievable, and the release documentation to state all material limitations. A null Stage 08 result is an acceptable scientific outcome; an integrity or reproducibility failure is not.
