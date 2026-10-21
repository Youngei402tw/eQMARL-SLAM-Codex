# Locked confirmatory results

Each cell is a mean across independent trained seeds. Test policies are frozen and greedy; the critic is not executed during deployment.

| Method | Seeds | Physical coverage | Map accuracy | Success | ATE (m) | Collisions | Bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| central_classical | 8 | 0.94868 | 0.95764 | 1.00000 | 0.08200 | 0.00000 | 10050.38625 |
| quantum_noent | 8 | 0.94608 | 0.96097 | 1.00000 | 0.08360 | 0.00750 | 10736.32875 |
| eqmarl | 8 | 0.95038 | 0.95875 | 1.00000 | 0.08392 | 0.00000 | 9788.69875 |

Paired differences (first minus second);95% percentile intervals resample paired trained seeds:

- eqmarl versus quantum_noent, physical_coverage: 0.004305555555555521, interval [-0.014583333333333351, 0.026145833333333326]; 8 paired seeds.
- eqmarl versus quantum_noent, success: 0.0, interval [0.0, 0.0]; 8 paired seeds.
- eqmarl versus central_classical, physical_coverage: 0.0017013888888888495, interval [-0.01614583333333336, 0.024270833333333255]; 8 paired seeds.
- eqmarl versus central_classical, success: 0.0, interval [0.0, 0.0]; 8 paired seeds.
- quantum_noent versus central_classical, physical_coverage: -0.0026041666666666713, interval [-0.021875000000000006, 0.014062499999999992]; 8 paired seeds.
- quantum_noent versus central_classical, success: 0.0, interval [0.0, 0.0]; 8 paired seeds.

Failed/incomplete runs: 0. All remain in the ledger.

This is a fixed short-budget2D landmark-SLAM experiment in a known empty workspace. No convergence, hardware speedup, general quantum advantage, or equivalence claim follows. Exact state-vector training is distinct from frozen-critic finite-shot sensitivity.
