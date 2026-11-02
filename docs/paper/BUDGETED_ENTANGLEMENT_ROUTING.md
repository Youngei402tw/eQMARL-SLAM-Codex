# Budgeted Entanglement Routing for Sequential Multi-Agent Reinforcement Learning

> A longer tutorial-style manuscript with diagrams, terminology, a worked example, full task definition, and interpretation is available in [BUDGETED_ENTANGLEMENT_ROUTING_EXPLAINED.md](BUDGETED_ENTANGLEMENT_ROUTING_EXPLAINED.md).
>
> Publication artifacts: [LaTeX source](BUDGETED_ENTANGLEMENT_ROUTING.tex), [BibTeX references](BUDGETED_ENTANGLEMENT_ROUTING.bib), and [compiled PDF](BUDGETED_ENTANGLEMENT_ROUTING.pdf). Run `make` in this directory to rebuild the PDF.

## Abstract

Entanglement-assisted multi-agent studies often treat entanglement as an always-available property of a fixed pair. We study a different control problem: four decentralized agents face a changing graph of private-information coordination requests, while a shared controller must route six consumable Bell pairs over ten steps. Each allocation must be a matching, is chosen before private contexts arrive, and competes with future requests. We introduce BER-MARL, a masked shared-action DQN that observes public request values, graph structure, remaining budget, and horizon. Request-level classical values are obtained by exhaustive local-strategy enumeration, and Bell values are independently bounded by XOR-game semidefinite programs. In a locked evaluation with 16 training seeds and 1,000 paired held-out sequences per seed, BER-MARL obtained mean entanglement surplus `0.5064`, versus `0.3084` for myopic greedy, `0.2268` for random, and `0.2248` for static allocation. The paired BER-MARL-minus-greedy difference was `0.1980`, 95% bootstrap interval `[0.1914, 0.2043]`, and every seed was positive. BER-MARL recovered `88.66%` of the exact online dynamic-programming surplus. A preregistered one-sided lower-bound requirement of more than 90% was not met (`87.68%`), so the strongest near-oracle composite claim is rejected. The result establishes a reproducible benchmark and shows substantial learned allocation gains under zero-shot transition-catalog shift, while identifying near-oracle transfer as an open problem.

## Problem and contribution

At each time step, public pairwise requests connect four agents. Each endpoint privately observes one of three contexts and emits a binary action. A request succeeds when the action parity satisfies a type-specific `3 × 3` predicate. Context weights are nonuniform, and urgency multiplies reward. The allocator observes public request metadata but never sees private contexts or post-context messages.

One Bell pair lets a selected edge use its certified entangled local policy. Unselected edges use their exact optimal zero-message classical policy. Selected edges must form a matching and consume Bell pairs from an episode budget of six. The public request graph evolves as a Markov process over a ten-step horizon.

The contribution is the combined sequential resource problem and its controlled evaluation:

1. a hash-locked benchmark with heterogeneous three-input request games, including zero-gap games;
2. exact local-classical values and independent SDP bounds for every request policy;
3. BER-MARL, which learns long-horizon allocation over feasible matchings;
4. paired comparisons with random, static, greedy, exact online DP, future-clairvoyant, and communication-enabled references;
5. retained raw results, failed runs, checkpoints, preregistered inference, and an independent release audit.

This is an algorithmic resource-allocation result. It does not introduce a new Bell inequality and does not claim quantum computational speedup or hardware advantage.

## Method

BER-MARL uses a masked DQN with two 64-unit hidden layers. Its input contains six current edge gains, six activity bits, remaining Bell-pair budget, and remaining horizon. Its ten global actions are the empty matching, six single edges, and three disjoint two-edge matchings; infeasible actions are masked. Training uses Adam at `5e-4`, replay batches of 64, a target update interval of 200 steps, and 2,500 episodes per seed. Sixteen final seeds train on development and validation catalogs. Final checkpoints are evaluated without adaptation on a sealed request-type family, a sealed public transition catalog, and 1,000 paired sequences.

The primary outcome is entanglement surplus: episode return minus exact classical zero-message return on the same requests. The unit of inference is the independent training seed. Twenty thousand seed-level bootstrap resamples produce the registered intervals.

## Results

| Method | Mean entanglement surplus |
|---|---:|
| Classical exact | 0.00000 |
| Quantum static | 0.22478 |
| Quantum random | 0.22685 |
| Quantum greedy | 0.30839 |
| BER-MARL | **0.50642** |
| Exact online DP | 0.57122 |
| Future-clairvoyant | 0.58326 |

BER-MARL exceeded greedy by `0.19803` surplus, with paired 95% interval `[0.19141, 0.20432]`. This is a 64.2% increase over greedy surplus. BER-MARL captured 88.66% of online-DP surplus. Its one-sided 95% lower bound was 87.68%, below the preregistered 90% requirement. Four of five composite gates passed; only the near-oracle fraction gate failed. No budget or matching violation occurred.

The 12 held-out request games had Bell-minus-classical gaps between `0` and `0.03830`, mean `0.01760`. This heterogeneity is essential: a router must avoid spending scarce pairs on zero-gap and low-value requests while retaining capacity for urgent future opportunities.

## Interpretation and limitations

The data support consistent learned improvement over myopic allocation on unseen request types and unseen public dynamics. They do not support the stronger claim that zero-shot BER-MARL is reliably above 90% of a model-aware DP oracle. The gap isolates a concrete next question: whether online public-transition estimation or meta-RL can close the remaining 11.34% without exposing private contexts.

All results are analytic simulation. Bell policies use two-qubit ideal correlations; no physical shots or hardware noise were evaluated in the confirmatory study. The message oracle is a privileged reference and is not resource-comparable. The bounded novelty screen supports a plausible literature gap, but a submission still needs a broader human literature review before making any priority claim.

## Reproducibility

The prospective plan is in `docs/BUDGETED_ENTANGLEMENT_MARL_PLAN.md`. Stage reports 20–24 document every gate and resource budget. The authoritative confirmatory bundle is `experiments/23/23-budgeted-entanglement-confirmatory_attempt-01`; the independent audit is `experiments/24/24-budgeted-entanglement-release-audit_attempt-01`. Raw episode rows, resolved configurations, request certificates, checkpoints, bootstrap outputs, artifact hashes, and retained failures are committed on their stage branches.
