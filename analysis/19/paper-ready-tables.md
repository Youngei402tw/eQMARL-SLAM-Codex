# Paper-ready nonlocal QMARL tables

## Frozen-policy exact performance

| Method | Mean win probability | Minimum | Maximum | Information/resource class |
| --- | ---: | ---: | ---: | --- |
| Classical local | 0.749925 | 0.749916 | 0.749932 | private local bit only |
| Classical shared randomness | 0.749970 | 0.749970 | 0.749971 | private local bit + pre-input shared randomness |
| Quantum separable | 0.749999 | 0.749996 | 0.750000 | matched local quantum measurements, no entanglement |
| Quantum entangled | **0.853553** | 0.853553 | 0.853553 | one Bell pair, local measurements, no communication |

Sixteen fresh paired training seeds were used. All values are final-checkpoint exact expectations under uniform CHSH contexts.

## Preregistered primary inference

| Quantity | Result |
| --- | ---: |
| Entangled − strongest paired admissible control | **0.10355394** |
| Two-sided 95% paired-seed bootstrap interval | **[0.10355339, 0.10355476]** |
| Entangled mean | **0.85355339** |
| One-sided 95% lower bound | **0.85355339** |
| Certified classical no-communication ceiling | 0.75000000 |
| Prespecified meaningful effect over ceiling | 0.05000000 |
| All registered positive gates | **Passed** |

## Finite-shot and noise results

| Condition | Win probability/rate | 95% interval |
| --- | ---: | ---: |
| 131,072 fresh simulated entangled-policy shots | 0.852547 | [0.850617, 0.854456] |
| Exact, no depolarizing noise | 0.853553 | — |
| Exact, 2% local depolarizing noise | 0.839553 | — |
| Exact, 5% local depolarizing noise | 0.819082 | — |

The noise rows are exact simulator sensitivity results. They are not hardware measurements.
