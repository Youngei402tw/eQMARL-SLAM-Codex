# 05-frozen-pilot-replay-v1_attempt-01

Hypothesis: Every saved pilot policy reproduces its registered greedy validation outcome and exposes action/reward counts without critic execution.

Outcome: completed. Gates passed: True.

Raw records and diagnostics are authoritative. One fixed validation-map replay per saved pilot. Detailed action/duration/reward counts describe these frozen replays; original training action sequences were not retained. No optimizer updates, critic executions or final-test access. No physical quantum shots were sampled. Failed outcomes remain archived.

Reproduce from source 6d3b7724e1eb89c81d9541706e72aaed578eebba with the committed config after publishing a fresh source receipt; use a new attempt ID to retain this bundle. Verify with `python -m eqmarl_slam.cli verify --run experiments/05/05-frozen-pilot-replay-v1_attempt-01`.
