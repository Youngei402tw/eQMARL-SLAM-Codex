# 09-release-audit-v1_attempt-01

Hypothesis: A clean locked runtime can load every frozen confirmatory checkpoint, exactly replay one validation scenario per method, and regenerate the locked analysis byte-for-byte without training.

Outcome: failed during post-execution archive validation. The read-only diagnostic gates passed, but the resolved model metadata omitted the schema-required `encoder` field.

Raw records and diagnostics are retained. This was a release packaging failure after the checks ran; it performs no training, no test-set policy selection, and no new scientific comparison. No physical quantum shots were sampled.

Reproduce from source e21c6bbd16ff7d307418f8e0588b440438a0237f with the committed config after publishing a fresh source receipt; use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run experiments/09/09-release-audit-v1_attempt-01`.
