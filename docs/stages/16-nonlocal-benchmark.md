# Stage 16 — nonlocal benchmark and correctness

## Preflight

The stage asks whether the smallest exact implementation supports the registered opportunity before any model selection. The falsifiers were an incorrect classical ceiling, incorrect Bell probability, information leakage, a separable control crossing the local ceiling, bad gradients, or sampled behavior inconsistent with the exact probability.

The registered source was published at `955571b87a07562aa6d6a9ea344492a8c4f2c73b` before execution. The dependency is the published Stage 15 completion commit `519d75e8f82369edfa580e62fe52558c84595159`.

## Result

All seven gates passed:

- exhaustive enumeration of all 16 deterministic local strategy pairs gave maximum expected reward exactly `0.75`; eight strategies attain it;
- the Bell policy reached `0.8535533905932737`, exactly `cos(pi/8)^2` to the registered tolerance;
- four independent PennyLane circuit correlations agreed with the analytic expression within `3e-16`;
- the separable quantum policy reached but did not exceed `0.75` across the deterministic candidate and 32 optimization starts;
- analytic gradients matched finite differences with maximum absolute error `7.70e-11`;
- 65,536 sampled decisions produced win rate `0.8554993`, 1.409 standard errors from the exact probability;
- each actor input contained only its own private bit, with zero post-input communication.

The full repository test suite passed after implementation: 120 tests, with two PennyLane deprecation warnings.

## Resources and interpretation

The diagnostic consumed 65,536 simulated decisions, 12,800 separable-bound optimizer updates, four analytic circuit fixture evaluations, zero physical shots, 7.878 experiment wall-seconds, and 0.00219 CPU-core-hours. The stage used a conservative additional 300 seconds for implementation and test setup.

This verifies the benchmark and makes a positive learned-policy result possible. It is not itself a training result or a quantum-advantage claim. Stage 17 must use development-only seeds and freeze the confirmatory protocol before Stage 18.

## Gate

**Passed.** The complete bundle is `experiments/16/16-nonlocal-correctness_attempt-01` and validates against the repository archive schema.
