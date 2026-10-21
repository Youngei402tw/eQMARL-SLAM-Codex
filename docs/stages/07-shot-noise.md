# Stage 07 — Finite shots and a declared noise channel

Status: passed. Question: how sensitive are the matched fitted analytic critics to shot sampling and one explicit channel model? The smallest falsifiers are failed seeded replay, an incompatible differentiation path, or failure of the exact channel identity.

Freeze the archived matched fitted critics and four fixed validation observations. Use64/256/1024shots with eight independent replicates, record predictions and full encoder/circuit/head parameter-shift Jacobians against the same analytic one-observation loss. Value and gradient errors are descriptive; relative joint encoder/circuit gradient error(excluding affine head)>0.5 is flagged unreliable, not silently removed. The declared noise grid is0,0.01,0.05,0.15 independent single-qubit depolarization after all unitaries. Mixed-state analytic parity must scale by(1-4p/3)^4 before the affine bias. Finite noisy readouts at256shots are separate from exact mixed-state expectations.

This is frozen fitted-critic sensitivity, not finite-shot policy training or a hardware/network claim. No optimizer updates occur. Exact simulator shots=not physically sampled; finite simulator sampling is charged to actual device executions and shots including backward, seeded checks and all repeats. Estimated<10000 calls and<2million simulated shots, memory<1GB, family cap1h, stage cap24active hours. The cap stops and retains a regime if exceeded; no new compute.

Exit: reproducible sampling, valid finite gradients, exact channel check, credible all-call accounting and remotely verified raw/checkpoint artifacts. Sampling or noise degradation is a valid negative result. Analytic training remains the scope of the locked task comparison.

Dependency completion: `0647dc61b203825119e414189355c70d8c5410a5`.

Preflight failure retained in `configs/stage07/preflight-channel-diagnosis.json`: mixed-backend zero-noise error1.47e-8 under Torch defaultfloat32, despite explicit double inputs. Setting the frozen diagnostic process defaultfloat64 restores agreement2.22e-16. No actor is initialized/trained in this process; the analytic training path retains its prior initialization semantics. The mixed API now rejects an incorrect process default instead of silently losing precision. Original discrepancy source is retained in Git.

All seeded replay, finite parameter-shift and exact channel gates passed. All replicates remain archived.

| Critic | Shots | Physical-value RMSE | Relative error of mean gradient | Single-replicate gradient RMS |
|---|---:|---:|---:|---:|
| quantum_noent | 64 | 0.680266 | 0.066861 | 0.231938 |
| quantum_noent | 256 | 0.346200 | 0.050413 | 0.127833 |
| quantum_noent | 1024 | 0.139883 | 0.024853 | 0.057057 |
| eqmarl | 64 | 0.470510 | 0.086698 | 0.234896 |
| eqmarl | 256 | 0.234710 | 0.040483 | 0.115680 |
| eqmarl | 1024 | 0.120598 | 0.026533 | 0.053382 |

Gradient errors above exclude affine-head parameters; group-level and full-vector results are retained. The original immutable field `gradient_relative_rmse` computes relative error of the mean of eight gradient estimates. It is not RMS across individual estimates; the [raw-derived clarification](../../analysis/07/shot-analysis.json) records both. No mean-gradient setting crossed the registered0.5 threshold. Four observations and one gradient input do not establish reliable task-policy training.

Value and gradient sampling error generally decrease with shots in this fixed sample. The declared final-wire noise increasingly biases the frozen readout; exact channel identities hold to numerical precision. This channel omits gate-dependent errors, distribution loss, topology and hardware timing, and cannot support a full hardware/network robustness claim. No finite-shot/noisy task policy was trained; the subsequent confirmation remains analytic.

Source `feef26a272dfa658176e8f4c9ef3b8df8f67faf8`; results `fd25b5213a3cd6b3d52ee17271bd5bab52246d99`. All134 raw records, resolved configuration, diagnostics and referenced frozen checkpoints are permanently stored in Git. The run bundle was independently retrieved and all7 inventoried files verified; referenced checkpoints were already verified in Stage05 and are hashed again in this run. See [run index](../../experiments/07/index.json) and [publication receipt](07-publication.json).

Actual resources:3246 device executions,1378304 simulated measurement shots including backward,19.783sec wall,0.005495CPU core-hours;zero optimizer updates, simulator primitives and GPU hours. No physical hardware was used. The failed precision preflight remains in its original Git commit and diagnosis JSON; it was resolved before this sole registered run. Fifteen targeted quantum checks passed. Stage/family active usage is recorded in project state and remains within24h/1h.
