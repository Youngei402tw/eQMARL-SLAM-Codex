# Stage 17 — bounded nonlocal-policy development and freeze

## Registered design

Four policy families each received four learning-rate candidates and two development seeds per candidate, followed by four pilot seeds at the selected rate. Every run used the final checkpoint. The stage used exact common-return gradients so all methods optimized the same objective without trajectory-sampling variance. Confirmatory seeds, finite-shot streams, update limits, inference, and failure handling were frozen before Stage 18.

Source commit `4ad54f488794bc05c9a191651c8cda88c5c69f2f` was published and verified before execution. The dependency is Stage 16 completion `235afac5d0dba03c1c84184905459cbe3a59f92e`.

## Development result

Selected learning rates were `0.3` for both classical policies and `0.03` for both quantum policies. Four-seed pilot mean final expected win probabilities were:

| Method | Mean | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| Classical local | 0.749927 | 0.749920 | 0.749939 |
| Classical shared randomness | 0.749971 | 0.749970 | 0.749971 |
| Quantum separable | 0.749999 | 0.749997 | 0.750000 |
| Quantum entangled | 0.853553 | 0.853553 | 0.853553 |

All gradients were finite, serialized policies replayed exactly, every control respected the 0.75 ceiling, and every entangled pilot exceeded the 0.82 feasibility threshold.

## Confirmatory freeze

Stage 18 has 16 fresh paired seeds (`18001`–`18016`), 500 final-checkpoint updates, and the four registered methods. The primary comparator is the per-seed maximum of shared-randomness classical and separable quantum controls. A positive claim requires all three conditions:

1. the two-sided 95% paired bootstrap interval for entangled minus comparator has lower bound above zero;
2. the one-sided 95% seed-bootstrap lower bound for the entangled mean exceeds 0.75;
3. the entangled mean exceeds 0.75 by at least the prespecified 0.05 meaningful-effect threshold.

Bootstrap uses 10,000 resamples and RNG seed 19001. Secondary operational evaluation uses 8,192 fresh finite-shot decisions per seed with locked seeds `28001`–`28016`. Local depolarizing probabilities 0, 0.02, and 0.05 are descriptive secondary conditions. No missing seed may be replaced.

The machine-readable freeze is `experiments/17/17-nonlocal-development_attempt-01/confirmatory_freeze.json`.

## Resources and gate

The stage completed 32 tuning runs and 16 pilots, totaling 20,800 optimizer updates, 41,600 circuit-equivalent context evaluations, zero sampled shots, 9.587 experiment wall-seconds, and 0.00266 CPU-core-hours. A conservative 400 seconds is charged for implementation and testing.

**Passed.** The experiment is capable of testing the prospective claim, and no confirmatory outcome has been accessed.
