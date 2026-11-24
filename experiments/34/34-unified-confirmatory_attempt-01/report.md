# 34-unified-confirmatory_attempt-01

Hypothesis: Primary: on a sealed test split opened once, with every method fixed beforehand, the selected PPO learner minus the validation-tuned bid-price rule has a seed-bootstrap 95% interval entirely below zero (bid-price superior). Secondary: PPO superior to greedy; DQN versus bid-price; PPO-DQN equivalence within 5% of the online-DP mean.

Outcome: completed. Gates passed: True.

Raw records and diagnostics are authoritative. Single confirmatory evaluation of all methods on a test split opened exactly once. No physical quantum shots were sampled. Failed outcomes remain archived.

Reproduce from source fd6744e138fe8cf97fc843b4252f5bb6b8bc753a with the committed config after publishing a fresh source receipt; use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run experiments/34/34-unified-confirmatory_attempt-01`.
