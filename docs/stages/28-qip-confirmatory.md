# Stage 28 — locked robust entanglement-routing evaluation

## Preflight

**Question.** On fresh request games and public dynamics, including unseen six-agent ring and eight-agent sparse graphs, does validation-selected PPO improve over greedy allocation and how does it compare with the strongest conventional DQN comparator?

**Falsifiers.** PPO superiority over DQN requires a seed-paired 95% bootstrap lower bound above zero. PPO and DQN equivalence requires the paired 90% interval to lie wholly within ±5% of the online-reference surplus. PPO improvement over greedy requires a seed-level 95% lower bound above zero. These outcomes are independent; every null or negative result will be retained.

**Locked design.** Train twelve fresh PPO and twelve fresh DQN policies for 2,500 episodes on combined development/validation catalogs. Evaluate final checkpoints on 500 paired sequences for each of four sealed catalogs. Classical exact, random, static, greedy, online model-aware, future-clairvoyant, and message oracles are reported with their information differences. Four frozen noisy-resource conditions use 100 paired sequences per catalog for PPO to remain within the 2.5-million-decision cap; this secondary count is fixed before opening test values.

**Gate.** Complete all 24 models, all primary and secondary evaluations, 20,000 bootstrap resamples, Holm-adjusted catalog contrasts, and safety checks within four active wall-hours and 2.5 million decisions. A scientific null result passes the execution stage when complete and correctly reported.

## Confirmatory results

All 24 frozen final-checkpoint policies completed on all four sealed catalogs. Aggregate mean entanglement surplus was:

| Method | Mean surplus |
|---|---:|
| Classical exact | 0.00000 |
| Random | 0.22632 |
| Static | 0.23539 |
| Greedy | 0.41828 |
| PPO | 0.69463 |
| DQN | **0.70648** |
| Online model-aware reference | 0.80600 |
| Future-clairvoyant reference | 0.84264 |

PPO did not outperform DQN. The paired seed mean difference was `-0.011849`, with 95% bootstrap interval `[-0.026495, 0.000376]`. The registered practical-equivalence test did pass: the two-sided 90% interval `[-0.023958, -0.001137]` lay inside the fixed margin `±0.040300`, equal to 5% of online-reference surplus. Thus the held-out evidence supports optimizer-robust resource-aware learning, not PPO superiority.

PPO exceeded greedy by `0.276342`; its seed-level 95% interval was `[0.268796, 0.282980]`. DQN and PPO recovered `87.65%` and `86.18%` of the online-reference surplus, respectively. Neither method was uniformly best by catalog. No PPO-versus-DQN catalog contrast survived Holm adjustment at `0.05`.

The eight-agent sparse catalog produced the largest absolute surplus, with PPO `1.14338` and DQN `1.16749`. This establishes execution and transfer through eight agents within the registered sparse topology, but does not establish asymptotic scaling.

## Resource conditions and limitations

PPO mean surplus was `0.51342` at heralded success `0.75` and `0.32983` at `0.5`. It was exactly zero at Werner visibility `0.8` and `0.6`. The sixteen held-out request types required visibility from `0.8491` to `1.0` for positive Bell-minus-classical gain, with median `0.9620`. The main physical limitation is therefore the high fidelity required by this request family.

The message oracle reached `20.03315`, but it exchanges private inputs and solves a stronger-information problem. It is not an allocator comparator and is excluded from learned-policy superiority claims.

## Resources and gate

The run consumed 2,081,152 decisions, 556,556 total optimizer/certification updates, 1,843,200 circuit-equivalent analytic evaluations, 8,323.98 wall-seconds, and 2.31203 CPU-core-hours. It sampled zero physical shots. All 24 checkpoints, 81,200 episode rows, 88 diagnostic metric rows, request certificates, and null outcomes are retained. No budget or matching violation occurred.

**Passed.** All execution, completeness, safety, statistical-analysis, and archive gates passed. Scientifically, PPO superiority was null, PPO–DQN practical equivalence passed, and learned allocation substantially exceeded greedy under ideal entanglement. Moderate Werner-noise robustness failed.
