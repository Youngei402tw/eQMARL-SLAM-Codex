# Stage 18 — locked learned nonlocal quantum-advantage test

## Locked execution

The remotely verified source commit was `08a84f5f44ac8d6d9574152dad60eb5e1ae41156`; it consumed the immutable Stage 17 freeze from dependency completion `a9cb465b8ae29eea0492e1d9d95ba5bfdae1fbc9`. The matrix contained four methods and 16 fresh paired initialization seeds. Every model received 500 exact common-return updates and used the final checkpoint. No seed failed or was replaced.

## Primary result

All preregistered positive conditions passed. The learned entangled policy mean was `0.8535533906`. The strongest paired admissible control, chosen conservatively as the larger of shared-randomness classical and separable quantum for each seed, yielded an entangled-minus-control estimate of `0.1035539434`; the two-sided paired-seed bootstrap interval was `[0.1035533935, 0.1035547580]`.

The one-sided 95% seed-bootstrap lower bound for the entangled mean was `0.8535533906`, above the exact classical no-communication ceiling `0.75`. The effect over the ceiling was `0.1035533906`, exceeding the prespecified meaningful-effect threshold `0.05`.

| Method | Mean exact win | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Classical local | 0.749925 | 0.749916 | 0.749932 |
| Classical shared randomness | 0.749970 | 0.749970 | 0.749971 |
| Quantum separable | 0.749999 | 0.749996 | 0.750000 |
| Quantum entangled | **0.853553** | 0.853553 | 0.853553 |

## Finite-shot and noise evidence

Fresh finite-shot PennyLane evaluation produced 111,745 wins in 131,072 entangled-policy decisions: rate `0.8525467`, Wilson 95% interval `[0.8506169, 0.8544558]`. This interval remains wholly above 0.75.

Registered exact local-depolarizing sensitivity gave mean rewards `0.8535534` at 0%, `0.8395527` at 2%, and `0.8190819` at 5%. These are simulator conditions, not hardware measurements.

## Resources, artifacts, and claim

The complete bundle contains 64 final checkpoints, the resolved config, freeze identity, 131 raw metric rows, diagnostics, finite-shot outcomes, noise rows, analysis, and an artifact inventory. It validates 72 files.

Execution consumed 524,288 sampled decisions across all methods, 32,000 optimizer updates, 64,128 circuit or circuit-equivalent evaluations, 262,144 simulated quantum measurement shots, 14.656 wall-seconds, and 0.00407 CPU-core-hours. A conservative 500 seconds is charged for source-lock implementation and review.

The registered result supports the phrase **learned quantum advantage** only for a two-agent decentralized CHSH contextual bandit under the no-post-input-communication contract and analytic/sampled simulation. It does not establish computational speedup, physical hardware advantage, active-SLAM advantage, or usefulness when classical communication is available.

## Gate

**Passed with a positive result.** Stage 19 must reproduce the analysis independently, validate checkpoint and archive identities, and ensure the paper framing retains every scope limitation.
