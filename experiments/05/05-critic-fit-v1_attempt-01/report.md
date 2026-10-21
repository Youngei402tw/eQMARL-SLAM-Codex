# 05-critic-fit-v1_attempt-01

Hypothesis: Matched split circuits differ only by input entanglement; every method has meaningful gradients and signed-value capacity on a frozen real-return corpus.

Outcome: completed. Gates passed: True.

Raw records and diagnostics are authoritative. Corrected common-head component fitting; actor weights frozen only for the separately registered corpus. RL actors later train independently. Underfitting is reported, not silently repaired by enlarging one method. No physical quantum shots were sampled. Failed outcomes remain archived.

Reproduce from source 614879ad91f971e2ab55177cbd83d32f13f75512 with the committed config after publishing a fresh source receipt; use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run experiments/05/05-critic-fit-v1_attempt-01`.
