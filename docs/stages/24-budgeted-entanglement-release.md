# Stage 24 — budgeted-entanglement MARL release audit

## Audit result

All eight independent release gates passed. The audit revalidated the authoritative Stage 21–23 bundles, enumerated the exact classical optimum for all 12 held-out request games, independently resolved their XOR-game SDP ceilings, verified that every recorded Bell value stayed within numerical tolerance, rebuilt the method means and bootstrap intervals from raw episode rows, and hashed all 16 allocator checkpoints.

The independent reconstruction matched every reported scalar within `1e-12`. It preserved the positive paired BER-MARL-versus-greedy interval `[0.191412, 0.204323]` and the failed DP-fraction lower bound `0.876840 < 0.90`.

The release explicitly permits the claim that BER-MARL substantially improved held-out surplus over random, static, and greedy allocation. It prohibits claims that the five-part composite gate passed, that the result is a computational speedup or hardware advantage, that the request games constitute new Bell inequalities, or that the work is globally first without a broader human literature review.

The repaired GitHub CPU workflow passed before the confirmatory release and now validates the extended stage registry, reportable bundles, retained archive failures, and full test suite.

## Resources and gate

The audit read 23,000 raw episode rows and performed 12 independent request checks in 2.20 seconds with no training, circuit execution, or shots. A conservative 400 seconds covers implementation and verification.

**Passed.** The complete audit bundle is `experiments/24/24-budgeted-entanglement-release-audit_attempt-01`.
