# Stage 07 — Finite shots and a declared noise channel

Status: running. Question: how sensitive are the matched fitted analytic critics to shot sampling and one explicit channel model? The smallest falsifiers are failed seeded replay, an incompatible differentiation path, or failure of the exact channel identity.

Freeze the archived matched fitted critics and four fixed validation observations. Use64/256/1024shots with eight independent replicates, record predictions and full encoder/circuit/head parameter-shift Jacobians against the same analytic one-observation loss. Value and gradient errors are descriptive; relative joint encoder/circuit gradient error(excluding affine head)>0.5 is flagged unreliable, not silently removed. The declared noise grid is0,0.01,0.05,0.15 independent single-qubit depolarization after all unitaries. Mixed-state analytic parity must scale by(1-4p/3)^4 before the affine bias. Finite noisy readouts at256shots are separate from exact mixed-state expectations.

This is frozen fitted-critic sensitivity, not finite-shot policy training or a hardware/network claim. No optimizer updates occur. Exact simulator shots=not physically sampled; finite simulator sampling is charged to actual device executions and shots including backward, seeded checks and all repeats. Estimated<10000 calls and<2million simulated shots, memory<1GB, family cap1h, stage cap24active hours. The cap stops and retains a regime if exceeded; no new compute.

Exit: reproducible sampling, valid finite gradients, exact channel check, credible all-call accounting and remotely verified raw/checkpoint artifacts. Sampling or noise degradation is a valid negative result. Analytic training remains the scope of the locked task comparison.

Dependency completion: `0647dc61b203825119e414189355c70d8c5410a5`.

Preflight failure retained in `configs/stage07/preflight-channel-diagnosis.json`: mixed-backend zero-noise error1.47e-8 under Torch defaultfloat32, despite explicit double inputs. Setting the frozen diagnostic process defaultfloat64 restores agreement2.22e-16. No actor is initialized/trained in this process; the analytic training path retains its prior initialization semantics. The mixed API now rejects an incorrect process default instead of silently losing precision. Original discrepancy source is retained in Git.
