# Stage 27 — robustness, scaling, and confirmatory freeze

## Preflight

**Question.** Does resource-aware allocation remain useful across non-final changes in graph size, topology, horizon, budget, entanglement visibility, and heralded generation success?

**Smallest falsifier.** Train fresh seeds for the four frozen Stage 26 configurations on development catalogs, then evaluate the unchanged policies on all validation catalogs. A collapse to greedy behavior under modest visibility `0.8` or heralded success `0.75`, invalid actions, or sensitivity confined to one topology falsifies a broad robustness explanation. Every null condition remains in the archive.

**Deliverables.** Twelve fresh checkpoints; paired condition/catalog summaries for DQN, Double DQN, PPO, and recurrent PPO; ideal greedy and model-aware references under every condition; parameter, training, and inference costs; positive-gain prevalence; and a hash-locked Stage 28 configuration.

**Budget.** Three seeds per method, 1,800 episodes per seed, 200 paired validation sequences per catalog and condition, at most 1.5 million decisions, and three active wall-hours. One job may use at most four CPU threads.

**Gate.** Complete every registered condition across three, four, and six agents and path, ring, star, and sparse-random graphs; retain all outcomes; remain within the cap; and leave the eight-agent/test request family unopened. Scientific robustness is reported rather than required for stage completion.

## Results

All twelve fresh policies and 416 learned/reference condition-catalog evaluations completed. Ideal-condition mean surplus aggregated equally over the four validation catalogs was:

| Method | Mean surplus |
|---|---:|
| DQN | 0.32198 |
| Double DQN | 0.32054 |
| PPO | **0.32729** |
| Recurrent PPO | 0.32251 |

PPO remained positive on every validation catalog: `0.12719` for the three-agent ring, `0.35075` for the four-agent sparse graph, `0.55294` for the six-agent path, and `0.27828` for the six-agent star. The corresponding ideal greedy means were `0.10231`, `0.24867`, `0.27811`, and `0.23135`. This supports topology and size transfer through six agents, while the small number of catalogs limits generality.

Heralded generation failure produced gradual degradation. PPO mean surplus was `0.29803` at success probability `0.9`, `0.25257` at `0.75`, and `0.16347` at `0.5`, compared with `0.32729` ideally. Other methods followed the same pattern.

Werner visibility exposed a sharp limitation. PPO produced zero entanglement surplus at visibility `0.9` and below; DQN and Double DQN retained only `0.01012` and `0.01392` at `0.9`, then zero at `0.8`. This is caused primarily by the request family rather than allocator collapse: only `9.52%` of validation edge-state opportunities retained positive Bell-minus-classical gain at `v=0.9`, and none did at `v=0.8`. Across all 30 non-test request types, the analytic visibility threshold for positive gain ranged from `0.8511` to `1.0`, with median `0.9824`. The submission must present high entanglement fidelity as a central limitation.

## Resources, freeze, and gate

The run consumed 1,279,760 decisions, 126,534 optimizer updates, 2,979.00 active wall-seconds, and 0.82741 CPU-core-hours. It sampled no physical quantum shots. All twelve checkpoints, 428 raw records, diagnostics, and the complete Stage 28 freeze are retained. The freeze SHA-256 is `ff34ffbb5a30053555de83494c937980ad4f91c5e4e87f2401833cf173df60ef`.

**Passed with a failed broad noise-robustness hypothesis.** Completeness, scaling, topology, resource, safety, and test-sealing gates passed. Robustness to heralded loss was supported within the tested range; robustness to moderate Werner noise was not.
