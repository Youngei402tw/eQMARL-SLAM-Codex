# 32-learner-strength_attempt-01

Hypothesis: With a wider grid (past the previous edge), 2.4-3.2x longer training, a calibrated PPO value baseline, and 6 seeds, the best learner still trails the validation-tuned bid-price rule on fresh validation sequences of BellRoute v1.

Outcome: completed. Gates passed: True.

Raw records and diagnostics are authoritative. Reviewer-risk check on non-test data only; it cannot change any locked decision. No physical quantum shots were sampled. Failed outcomes remain archived.

Reproduce from source d66cd394107162f5b30d4d46eb2af5cbd5672e07 with the committed config after publishing a fresh source receipt; use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run experiments/32/32-learner-strength_attempt-01`.
