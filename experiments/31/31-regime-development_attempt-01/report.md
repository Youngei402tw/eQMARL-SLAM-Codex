# 31-regime-development_attempt-01

Hypothesis: On the hidden-regime variant, the best equally tuned learner (PPO, recurrent PPO, or DQN) is not futile relative to the strongest validation-tuned bid-price rule (within 5% of the belief-state DP mean on validation).

Outcome: completed. Gates passed: True.

Raw records and diagnostics are authoritative. Development on non-test data; the qip_v2 test split is hash-referenced but never opened. No physical quantum shots were sampled. Failed outcomes remain archived.

Reproduce from source 5d1885f4e55f443c2fad75a67bdb01ada3d05010 with the committed config after publishing a fresh source receipt; use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run experiments/31/31-regime-development_attempt-01`.
