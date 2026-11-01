# Stage 21 — certified request family and sequential environment

## Result

The certified task family is suitable for the planned resource-allocation study. Exhaustive enumeration valued every classical zero-message request policy, and independently solved XOR-game semidefinite programs bounded the Bell-pair policies. Across eight development and eight validation request types, the optimized Bell values met their SDP ceilings within `2.94e-10`.

The request-level quantum surplus was heterogeneous: it ranged from exactly `0` to `0.0514728`, averaged `0.0166135`, and exceeded `0.005` for 10 of 16 types. Retaining zero-gap types makes Bell-pair allocation a real selection problem instead of a rule that entangles every request.

The four-agent, ten-step Markov environment passed all exact budget and matching invariants over 100 diagnostic sequences. With six consumable Bell pairs, the exact online dynamic-programming policy obtained positive surplus on every sampled sequence. Mean oracle surplus was `0.56646` on the development catalog and `0.35755` on validation. The allocator sees only public graph, type, urgency, horizon, and budget fields; it never receives private endpoint contexts.

All nine registered gates passed. The PennyLane two-qubit fixture agreed with the analytic Bell correlations within `3.33e-16`. The final test request and catalog hashes remained sealed.

## Retained failures and resources

Attempts 01 and 02 passed every scientific gate but failed final archive validation because the first omitted the schema-required unused `train` field and the second incorrectly aliased a development-identity manifest. Both complete bundles are retained. Attempt 03 records the unused field as `null`, passed the same scientific gates, and validates fully.

Across all attempts, the stage consumed 3,000 diagnostic sequence decisions, 614,400 local-policy optimizer updates, 5,529,627 circuit-equivalent analytic evaluations, zero physical shots, 452.75 experiment wall-seconds, and 0.12574 CPU-core-hours. A conservative 500 seconds covers setup and testing.

## Gate

**Passed.** The authoritative bundle is `experiments/21/21-budgeted-entanglement-task_attempt-03`. This stage certifies the benchmark; it makes no learned-allocation claim.
