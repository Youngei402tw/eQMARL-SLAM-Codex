# 33-transition-robustness_attempt-01

Hypothesis: H1: a validation-tuned bid-price rule keeps >= 90% of online DP under every registered transition-matrix variant. H2 (expected to fail for structured dynamics, per an unregistered development-catalog sweep): CE-DP with the generic default guess and no logged data keeps >= 98% under every variant. H3: where H2 fails, CE-DP with 100 logged episodes recovers >= 98%.

Outcome: completed. Gates passed: True.

Raw records and diagnostics are authoritative. Generalization check on non-test data only; it cannot change any locked decision. No physical quantum shots were sampled. Failed outcomes remain archived.

Reproduce from source 07ecbbf5d265a8b37cf6106c45c70ee6c3ef68a2 with the committed config after publishing a fresh source receipt; use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run experiments/33/33-transition-robustness_attempt-01`.
