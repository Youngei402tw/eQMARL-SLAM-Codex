# Stage 19 — nonlocal QMARL reproduction and paper release

## Audit result

The independent release run validated the full Stage 18 bundle, replayed all 64 final checkpoints, matched every raw cell, regenerated the paired bootstrap estimate and interval exactly, and matched the finite-shot win and decision totals. All seven audit gates passed. The audit used no training, new task decisions, circuit executions, or shots; it consumed 1.070 wall-seconds and 0.000297 CPU-core-hours.

The complete repository suite passed 124 tests with two PennyLane deprecation warnings.

## Paper contribution

The release frames a boundary result: entangled centralized critic features did not improve active-SLAM coverage, while an entangled decentralized actor achieved a certified advantage in a nonlocal coordination task. The positive result is 0.853553 exact mean reward versus a 0.75 classical ceiling and 0.103554 over the strongest paired trained control. Fresh finite-shot rate was 0.852547 with interval [0.850617, 0.854456].

The paper explicitly identifies CHSH as established and claims novelty for the prospective QMARL methodology, matched attribution controls, negative-plus-positive mechanism contrast, learned-policy evidence, and auditable resource release. It does not claim hardware advantage, speedup, active-SLAM advantage, or general sequential-MARL superiority.

## Artifacts

- `docs/NONLOCAL_QMARL_PLAN.md`: prospective design
- `docs/NONLOCAL_QMARL_REPORT.md`: paper-ready narrative
- `analysis/19/paper-ready-tables.md`: concise tables
- `analysis/19/reproduced-analysis.json`: independent primary reconstruction
- `analysis/19/claim-limitations.json`: machine-readable claim audit
- `experiments/18/18-nonlocal-confirmatory_attempt-01`: raw confirmation, checkpoints, and analysis
- `experiments/19/19-nonlocal-release-audit_attempt-01`: clean audit bundle

## Gate

**Passed.** All planned outcomes and null/positive evidence remain archived. The release is complete once the results commit and a fresh remote retrieval are verified.
