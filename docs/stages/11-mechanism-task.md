# Stage 11 — Controlled complementarity task and diagnostics

Status: passed. Both attempts and all raw outputs are retained and independently verified from the remote branch.

## Intervention and contracts

The new `belief-controller-mechanism-v1` task preserves the Stage 08 two-robot simulator, GTSAM estimator, 27-feature recurrent actor, planner, communication, and PPO semantics. Landmark identities are divided deterministically into complementary parity groups. A robot contributes when it receives an informative local landmark factor from its assigned group. The normalized reward moves mass from two additive marginal events toward the joint event as complementarity alpha increases.

Across the balanced four-event counterfactual table, mean normalized reward is exactly 1 in every condition. Mixed interaction is 0.0 for additive, 1.333333 for low complementarity, and 4.0 for high complementarity. A local group cue is derived only from each robot's delivered landmark belief and is folded into the existing candidate-gain feature, retaining the actor dimension and information boundary.

The new `flexible_interaction` classical critic uses the same trainable 27-to-2 per-agent encoder and a symmetric MLP over latent sum, product, and absolute difference. It has 250 parameters, compared with 249 for the centralized classical control. All parameter groups received finite nonzero gradients and agent-order swap error was zero at `1e-12` tolerance.

## Attempts and results

Attempt 01 ran the registered checks but failed. The information-isolation probe modified simulator truth inside the scripted additive trajectory and did not restore it, so the reward-only dynamics gate correctly failed. Final archive validation also exposed that the global run schema enumerated only Stages 00–09. The complete failed bundle, checkpoint, 77 raw rows, and 16.886 seconds are retained.

Attempt 02 isolated the truth-leak probe in a separate environment and extended the schema for the already-authorized stages. It passed every gate:

- counterfactual reward scale matched;
- mixed interaction increased monotonically;
- actor truth isolation held;
- every reward decomposition summed and remained finite;
- identical scripted actions, primitive counts, and physical coverage occurred across reward conditions;
- joint complementarity events occurred in the actual SLAM task;
- the flexible classical critic was symmetric and fully connected to gradients;
- a 500-primitive eQMARL smoke trained with finite connected gradients;
- the final-test scenario catalog was not loaded by the diagnostic.

The measured quantum smoke rate was 0.027873 seconds per primitive. Combining that rate for 48 quantum models with the historical Stage 08 rate for 72 classical models, a 2× safety factor, and 15% evaluation overhead projects 61,815 seconds for the 120-model Stage 13 matrix. This is below the 86,400-second cap, so the preregistered 10,000-primitive common budget remains feasible.

## Resources and gate

Both attempts together used 1,102 primitives, 772 decisions, 44 optimizer updates, 1,058 analytic circuit evaluations, zero shots, 33.324 experiment wall-seconds, and 0.00925 CPU-core-hours. The conservative active stage envelope was 395.557 seconds, below the cap.

Fresh retrieval at results commit `12572494bbc693720bdd3f750fa253ac6445f63c` validated both bundles and every inventoried file. The receipt is `docs/stages/11-publication.json`.

Stage gate: passed. This establishes the task intervention and feasibility only; it provides no comparative policy-performance result.
