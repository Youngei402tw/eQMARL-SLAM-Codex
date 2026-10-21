# Final research report

## Result

The staged 2D cooperative active-SLAM study completed its full authorized plan. The locked confirmatory experiment retained all 24 trained models: centralized classical, matched quantum without input entanglement, and eQMARL, each with eight paired training seeds, 10,000 training primitives, and 100 held-out maps per final policy.

| Method | Seeds | Physical coverage, mean ± seed SD | Map accuracy | Success | ATE (m) | Communication bytes |
|---|---:|---:|---:|---:|---:|---:|
| centralized classical | 8 | 0.94868 ± 0.02573 | 0.95764 | 1.00000 | 0.08200 | 10050.39 |
| quantum, no input entanglement | 8 | 0.94608 ± 0.02253 | 0.96097 | 1.00000 | 0.08360 | 10736.33 |
| eQMARL | 8 | 0.95038 ± 0.01400 | 0.95875 | 1.00000 | 0.08392 | 9788.70 |

The preregistered primary eQMARL-minus-quantum-noent physical-coverage difference was 0.00431, with a paired trained-seed 95% percentile bootstrap interval of [-0.01458, 0.02615]. The eQMARL-minus-classical difference was 0.00170 [-0.01615, 0.02427]. All three methods achieved success 1.0 on the fixed test manifest.

The result does not establish eQMARL superiority. Because no equivalence margin was preregistered, the intervals spanning zero do not establish equivalence either. The scientific outcome is a complete bounded-budget null finding for this task and implementation.

## What the stages established

Stages 01–04 established the circuit/reference contracts, information-isolated exploration controls, full-history GTSAM Pose2/Point2 belief estimator, and common recurrent controller. Stage 05 compared six matched critic families under equal tuning and found substantial seed variability in the three-seed exploratory pilots. Stage 06 showed that the proposed local interaction component can represent the registered synthetic utilities, while matched classical forms often matched it; it was not an environment-level policy intervention. Stage 07 measured finite-shot and depolarizing-noise sensitivity of frozen fitted critics, including backward-pass shots, without claiming a finite-shot-trained policy or hardware result. Stage 08 performed the only locked held-out method comparison. Stage 09 reproduced checkpoint loads, frozen validation episodes, and raw analysis in a clean environment.

## Evidence and resources

The release contains 84 immutable experiment bundles. All planned IDs have outcomes. Four failed attempts are retained with their raw records and failure explanations. No inventoried file exceeds 10 MiB.

Across recorded experiment segments, including failed attempts, the archive contains 348,262 primitive steps, 247,472 decisions, 19,241 optimizer updates, 242,371 analytic/simulated circuit evaluations, 1,378,304 simulated shots, 7,171.420 wall-seconds, and 1.99185 CPU-core-hours. GPU device-hours and physical hardware shots were zero. Stage 08's actor-plus-SLAM test phase used 16,842 primitives, 11,258 decisions, 146.226 wall-seconds, and 0.04062 CPU-core-hours, with zero critic executions and optimizer updates.

Every Stage 08 checkpoint, resolved configuration, raw metric file, diagnostic, and inventory was independently retrieved from the remote branch and hash-verified. The clean Stage 09 environment loaded all 24 checkpoints, reproduced three registered validation episodes with zero reported metric error at `1e-8`, and regenerated the locked analysis byte-for-byte at SHA-256 `800c7d1e03bc828f4d8430f79515402ebbd4161a614f6e82aa566bcaa5648d56`.

The authoritative machine-readable sources are `analysis/08/confirmatory-analysis.json`, `analysis/09/release-audit.json`, and the per-run bundles under `experiments/`.

## Scope and limitations

The task is a short-budget, two-robot, 2D known-empty workspace with known landmark identities, noisy body-frame sensing, fixed candidate actions, and a Gaussian GTSAM belief approximation. The study does not cover occupancy mapping, ambiguous association, 3D perception, more than two agents, ROS/Gazebo deployment, or physical quantum hardware.

Quantum training used analytic four-qubit simulation. Simulated circuit evaluations and shots are accounting units, not hardware speed or cost evidence. Stage 06 used synthetic unitless component targets. Stage 07 used frozen critics rather than shot-trained policies. The confirmatory budget is 10,000 primitives per model and does not establish convergence. Eight trained seeds give limited precision, and the inference is conditional on the fixed 100-map test manifest.

Checkpoints contain model, optimizer, random-generator, and next-scenario state at new-episode boundaries. Exact mid-episode resume and an automatic training-resume CLI are not implemented. Original Stage 04/05 pilot training action trajectories were not retained; later Stage 08 runs include complete action counts, reward components, value/target moments, and gradient-group diagnostics.
