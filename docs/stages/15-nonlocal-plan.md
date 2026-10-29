# Stage 15 — prospective nonlocal QMARL plan and integrity audit

## Preflight

- **Question:** Can a learned entangled decentralized policy cross a certified classical no-communication ceiling?
- **Smallest falsifier:** exhaustive strategy enumeration and an independently implemented Bell-policy probability check. Failure of either bound blocks training.
- **Deliverables:** prospective plan, prior-result integrity reconstruction, stage registry, compute budget, claim and inference gates.
- **Exit gate:** the archived Stage 13 primary result reconstructs without a material discrepancy, and the new design is committed before implementation or development outcomes.
- **Failure gate:** any discrepancy that can change the Stage 13 primary estimate is retained, disclosed, and resolved before this extension advances.

## Prior-result integrity audit

A new script reconstructed the primary result directly from all 12,000 raw held-out episode rows without importing the Stage 13 or Stage 14 analysis. It inspected 120 bundles, verified 960 archived artifact hashes, confirmed identical test-scenario ordering across every cell, checked that each raw test row names the archived final checkpoint, and compared the diagnostic payload with the raw row payload.

The reconstruction exactly matched the published estimate `0.007083333333333386`, bootstrap interval `[-0.15597395833333325, 0.1802934027777777]`, and all eight seed interactions. No material discrepancy was found.

The first clean-suite attempt could not collect tests because the fresh Python 3.12 environment lacked the repository's optional runtime dependencies. This infrastructure failure was retained in the session record. A local isolated audit environment was created with CPU PyTorch, PennyLane, GTSAM, and pytest; the complete suite then passed: 116 tests, with two PennyLane deprecation warnings. The local environment is ignored and is not a scientific artifact.

## New study decision

The previous experiment is internally consistent but tests a narrow centralized critic intervention. The new protocol moves the quantum resource into decentralized execution, where Bell correlations expand the achievable policy set under a certified information constraint. The full hypothesis, controls, inference, resource limits, and claim boundaries are fixed in `docs/NONLOCAL_QMARL_PLAN.md` before Stage 16 implementation.

## Resources and gate

Stage 15 used no environment decisions, optimizer updates, circuits, or shots. A conservative 900 active wall-seconds is charged for planning and audit work. The stage gate **passes**. The branch was created from final main receipt `dbe2903c61ade83e17db7d0d94f9fecbe0abbc13`; Stage 16 must branch from this stage's published completion commit.
