# Contributions and limitations (QIP submission)

## Strongest defensible contribution

Model-free allocators learn **when and where** to spend scarce, consumable, certified Bell pairs, using public information only. Under a preregistered, test-locked protocol, they recovered about 86–88% of the model-aware dynamic-programming optimum. Myopic greedy spending recovered about 52%. They transferred to an unseen eight-agent graph. The effect is **algorithm-agnostic**: PPO and DQN are practically equivalent. It is driven by the **resource state**: remaining budget, remaining time, and the magnitude and location of gains. It is **bounded by fidelity**: no sealed request keeps a quantum gain below Werner visibility 0.849.

## Supported (with registered tests)

| Claim | Evidence | Source |
|---|---|---|
| Learned allocator beats greedy (fixed 4-agent graph) | +0.1980, 95% CI [+0.1914, +0.2043]; 16/16 seeds | Stage 23 |
| PPO beats greedy on sealed catalogs | +0.276, 95% CI [+0.269, +0.283]; 12/12 seeds | Stage 28, primary test 2 |
| PPO ≈ DQN within ±5% of online DP | 90% CI [−0.0240, −0.0011] ⊂ ±0.0403 | Stage 28, primary test 3 |
| Budget, time, gain magnitude, and gain location matter | Ablations reduce surplus by 33–56% | Stage 26 (development) |
| Heralded loss degrades surplus proportionally (no collapse) | 0.745× at p_h = 0.75; 0.479× at p_h = 0.5 | Stages 27–28 |
| Transfer to fully fresh 6-agent ring and 8-agent sparse graph | PPO−greedy +0.314 [+0.305, +0.322]; ~86% of DP | Stage 28, post hoc descriptive |

## Null or failed (retained and reported)

| Claim | Outcome | Source |
|---|---|---|
| ≥90% of online-DP surplus (one-sided lower bound) | **Failed**: lower bound 87.68% | Stage 23 |
| Double DQN beats strongest learner | **Null**: −0.0039 vs PPO | Stage 26 |
| Broad robustness to Werner noise | **Failed**: zero surplus at v ≤ 0.8 (all learners) | Stage 27 |
| PPO superior to DQN | **Null**: −0.0118, 95% CI [−0.0265, +0.0004] | Stage 28, primary test 1 |
| Per-catalog PPO > DQN after Holm | **Null** on all four catalogs | Stage 28, secondary |
| Surplus at v = 0.8 and v = 0.6 on sealed family | **Zero** | Stage 28, secondary |

## Not claimed

- No new Bell violation.
- No quantum computational speedup.
- No hardware evidence.
- No decentralized allocator: the allocator is centralized over public information, and only the endpoint games are distributed.
- No general quantum advantage beyond the certified XOR request family and the tested resource models.
- No PPO superiority.
- No near-oracle (90%) performance.

## Limitations reviewers will see

1. **Small quantum gaps.** The certified XOR games have gaps of a few percent, so the benefit vanishes under moderate depolarizing noise. High fidelity (v ≳ 0.85–0.95) is a precondition of the whole effect.
2. **Split overlap.** Two of the four sealed test catalogs reuse the public Markov dynamics of training catalogs, caused by a seed collision in the split generator. One validation catalog duplicates a development catalog. Request types are fresh, and transfer conclusions are also shown on the two fully fresh catalogs.
3. **Abstraction.** The model is single-hop with centralized delivery. It has no swapping, purification, or memory decoherence. Rewards are analytic expectations, no games are sampled, and nothing runs on hardware.
4. **Scale.** The study has 12 seeds, four test catalogs, at most eight agents, and 14 public states per catalog. The three-configuration tuning grid selected its largest learning rate.
5. **Implementation details.** The PPO value baseline is not scale-matched. The static baseline uses index-level priorities. The Codex runtime used torch 2.15 and numpy 2.5, not the pinned 2.14 and 2.4.
6. **Novelty.** No systematic literature review was performed, so the positioning is provisional.
