# Does task complementarity activate input entanglement in cooperative active SLAM?

## Contribution

This study turns the earlier eQMARL null into a prospective mechanism test. It introduces a controlled, belief-derived complementarity intervention whose scripted mixed interaction rises from 0.0 (additive) to 1.333 (low) to 4.0 (high) while reward scale, sensing, navigation, local information, actor architecture, and environment samples remain fixed. It compares the Bell input-entanglement critic with its matched no-entanglement control and three classical critics, including a 250-parameter symmetric interaction MLP matched closely to the 249-parameter centralized control.

The publishable contribution is a complete interaction phase diagram and falsification result: stronger registered complementarity did **not** reliably increase the downstream benefit of the tested input-entanglement intervention. This distinguishes “the original task lacked interaction” from “the tested entanglement mechanism did not exploit the controlled interaction under this learner and budget.” The released benchmark, task intervention, flexible classical control, paired inference, checkpoints, raw rows, and resource ledger let later methods test a sharper mechanistic hypothesis.

## Prospective design

The confirmatory matrix contained five critics × three complementarity conditions × eight paired trained seeds = 120 models. Each received 10,000 training primitives, 30 validation episodes, and 100 held-out actor-only test episodes from a hash-locked manifest. The final checkpoint was fixed before test evaluation. The primary seed-level estimand was:

`[coverage(eQMARL) - coverage(no-entanglement)]high - [coverage(eQMARL) - coverage(no-entanglement)]additive`.

The interval was a two-sided 95% percentile interval from 10,000 paired-seed bootstrap resamples. The registered minimum effect of scientific interest was 0.02 physical coverage. This was not an equivalence design.

## Result

The primary interaction was **+0.00708**, with 95% interval **[-0.15597, +0.18029]**. The estimate was below 0.02 and the interval spanned zero. The eight seed interactions ranged from -0.33722 to +0.46194, showing substantial between-training-seed variation.

Mean held-out eQMARL minus no-entanglement coverage was -0.07594 in additive, -0.01580 in low, and -0.06885 in high complementarity. In the high condition, eQMARL coverage averaged 0.72767, compared with 0.79653 for no-entanglement, 0.77878 for central classical, 0.81677 for bilinear, and 0.75642 for flexible interaction.

Descriptive high-condition eQMARL differences were -0.05111 versus central classical (95% unadjusted interval [-0.23185, 0.12153]), -0.08910 versus bilinear ([-0.21861, 0.03646]), and -0.02875 versus flexible interaction ([-0.10698, 0.05826]).

## Resource and integrity evidence

All 120 models and 12,000 held-out test episodes completed. The matrix used 1,501,711 total primitives, 1,242,090 agent decisions, 54,740 optimizer updates, 534,606 analytic circuit evaluations, zero physical shots, 27,929.729 experiment wall-seconds, and 7.757 CPU-core-hours. All raw outcomes, including the null, are retained.

The clean Stage 14 audit validated every bundle and checkpoint, reproduced all 15 representative method-condition validation paths exactly, and regenerated the primary estimate and interval exactly. The only post-lock analysis correction renamed a secondary field from `travel_distance` to the recorded `travel_distance_m`; it occurred before inference and did not change the primary metric, estimand, bootstrap, or data.

## Claims and limits

The evidence supports a scoped conclusion: in this two-robot analytic active-SLAM simulator, under the registered PPO learner, sample budgets, final-checkpoint rule, and critic definitions, increasing engineered complementarity did not reveal a reliable physical-coverage benefit from Bell input entanglement over the matched no-entanglement critic.

It does not establish equivalence, a general absence of useful quantum representations, hardware performance, speedup, larger-team scaling, or behavior under other learners and circuit families. Eight trained seeds leave a wide interval. The complementarity objective is engineered and belief-derived. Circuit execution used differentiable analytic simulation with zero physical shots. Strong classical results show why flexible classical interaction controls are necessary in future claims.

The earlier Stage 08 null remains an independent result. Together, the studies show no measured eQMARL coverage advantage in the original task and no reliable activation of that advantage after a prospective complementarity intervention.

Machine-readable inference is in `analysis/13/confirmatory-analysis.json`; paper tables and the claim matrix are in `analysis/14/`.
