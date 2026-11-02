# Stage 22 — BER-MARL development and confirmatory freeze

## Result

The bounded four-configuration search selected the 64-unit masked DQN with learning rate `5e-4`. The first attempt retained a failed bundle because an extra implementation gate required every deliberately varied tuning candidate to beat greedy, which was not a criterion in the prospective plan. It nevertheless showed all four combined-catalog pilots above the planned capability threshold.

Attempt 02 retained the selected configuration and strengthened evaluation by moving pilots to 300 public sequences disjoint from the model-selection stream. Every fresh pilot beat myopic greedy. Their entanglement surpluses were `0.31939` to `0.33014`, compared with greedy `0.23450` and exact online DP `0.34314`. The four DP fractions were `0.93079`, `0.94106`, `0.94409`, and `0.96213`, with mean `0.94452`.

The final test request values and public catalog were not opened. The full repository suite passed with 127 tests, and the repaired CPU workflow validates the 25-stage registry, all reportable bundles, and explicitly indexed archive-validation failures.

## Confirmatory freeze

`configs/stage22/stage23-freeze.json` fixes the selected architecture, 16 fresh training seeds, 2,500 episodes per seed, 1,000 paired held-out sequences, all eight comparison methods, 20,000 bootstrap replicates, and the five-part positive gate. The confirmatory allocator trains on the combined development and validation catalogs; only Stage 23 may certify and evaluate the sealed test family.

## Resources and gate

Across both retained attempts, Stage 22 consumed 344,000 decisions, 246,488 DQN updates, 522.81 experiment wall-seconds, and 0.14520 CPU-core-hours, plus a conservative 600 setup seconds.

**Passed.** Attempt 02 is authoritative. This is development evidence only and carries no held-out superiority claim.
