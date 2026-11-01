# Stage 23 — locked held-out entanglement-allocation study

## Held-out result

BER-MARL learned to conserve and route six Bell pairs across a ten-step, four-agent interaction graph on request types and public sequences absent from development. Across 16 fresh training seeds and 1,000 paired test sequences per seed, every learned allocator beat myopic greedy.

Mean entanglement surplus was:

| Method | Mean surplus |
|---|---:|
| Classical exact, no Bell pairs | 0.00000 |
| Quantum static | 0.22478 |
| Quantum random | 0.22685 |
| Quantum greedy | 0.30839 |
| BER-MARL | 0.50642 |
| Exact online DP | 0.57122 |
| Future-clairvoyant ceiling | 0.58326 |

BER-MARL improved over greedy by `0.198029`; the paired-seed 95% bootstrap interval was `[0.191412, 0.204323]`. The effect is 64.2% of greedy's surplus and 34.7% of the DP surplus. Its own surplus one-sided 95% lower bound was `0.500952`. No matching or budget violation occurred.

The twelve held-out request games had exact-classical-to-Bell gaps from `0` to `0.038296`, mean `0.017596`. Thus the result depends on selecting heterogeneous opportunities under scarcity, rather than applying entanglement to every request.

## Prespecified gate

Four of five conditions passed. BER-MARL recovered `88.66%` of online-DP surplus on average, with a one-sided 95% lower bound of `87.68%`; the preregistered condition required that lower bound to exceed `90%`. The run therefore does not support the plan's strongest near-oracle superior-allocation claim. It does support a narrower held-out finding: the learned zero-shot router consistently and substantially improves surplus over random, static, and greedy allocation under the registered resource and information constraints.

## Resources and retention

The run consumed 600,000 allocator decisions, 552,592 optimizer updates including held-out request certification, 1,382,400 circuit-equivalent analytic evaluations, zero shots, 956.91 wall-seconds, and 0.26579 CPU-core-hours. All 16 checkpoints, 16,000 learned-policy episode rows, baseline episode rows, request certificates, diagnostics, and the failed gate are retained.

## Gate

**Completed null result for the five-part composite gate.** The learned-versus-greedy primary comparison is positive, but the oracle-efficiency component failed. Stage 24 must audit and publish both facts.
