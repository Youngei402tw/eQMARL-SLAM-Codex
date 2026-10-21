# Stage 08 — Locked confirmatory comparison

Status: running; final-test outcomes have not been accessed. The complete machine-readable analysis plan is registered before training/testing.

Question: does eQMARL change paired seed-level held-out physical coverage relative to its matched no-input-entanglement critic under a fixed short learning budget? Centralized classical, selected by Stage05 validation return, is the required strong control. The candidate/rates use validation only; no new interaction or finite-shot task policy is selected.

Lock3 methods ×8 paired independent seeds8801–8808 ×10000 training primitives, followed by100 fixed paired held-out maps/disturbances per final policy. Common taskv2,16-unit trainable recurrent actor, local compression, planner/estimator, PPO/GAE, return normalization and final-checkpoint selection remain fixed. Greedy validation and test, sampled training. No optimizer/critic executes in final test; reload the serialized actor and verify frozen state/checkpoint hashes.

The provisional100000-step target projects96800seconds before final testing under conservative measured pilot throughput, exceeding24h. The common10000-step budget projects9680seconds and preserves all8 training replications and100 test maps. This confirms bounded-budget behavior, not convergence. One training job and one CPU thread, run cap3600sec, training cap3300sec, cumulative stage cap24active hours. No extension or new compute after test outcomes.

Primary: eQMARL minus matched noent in per-seed mean physical coverage. Average100 test episodes before inference; bootstrap paired trained seeds10000 times,95% percentile interval. Success and comparisons with classical are secondary/descriptive pointwise intervals. Inference is conditional on the fixed test manifest; episodes are not independent training replications. Nondetection is not equivalence. Report map accuracy, pose/trajectory error, collisions, communication and resources alongside coverage/success.

Keep every planned outcome. No bad-seed replacement, outcome-based checkpoint choice, test-time update or retuning. Entropy remains logged but cannot exclude a poor/deterministic policy; finite loss/gradient connections remain correctness gates. Failed/incomplete models are explicit ledger entries and make the full locked sample incomplete. Mechanical recovery must preserve source/protocol/budget and charge prior work; exact mid-episode resume is not claimed.

Exit: all planned outcomes archived, tables derived from raw hashes, no test leakage or outcome-driven changes, frozen-checkpoint verification and independent remote retrieval. The final test manifest has been reserved since Stage04 and is unchanged.
