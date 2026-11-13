# Stage 31 — Can learning beat strong rules when pacing is not enough? (proposal)

**Status: planned. No branch, no code, and no runs yet.** Execution needs explicit approval from the user, because it starts a new research direction and spends several CPU-hours on training.

## Why

Stage 30 showed that BellRoute v1 is essentially solved by simple rules:
- A two-parameter bid-price rule reaches 97.2% of the optimum.
- Certainty-equivalent DP (CE-DP) with no logged dynamics reaches 99.9%.

The learned allocators (86–88%) lose to both. A credible claim *for* learning therefore needs a variant in which:
1. the optimum, or a tight upper reference, is still computable;
2. the best **equal-information** simple rule leaves material headroom, measured *before* any learner is trained.

The equal-information rules are the bid-price family and other rules that see only what the learners see.

## Candidate variants

| Variant | Change | Why simple rules may fail | Optimal reference |
|---|---|---|---|
| **A. Generate-and-store (BellRoute-M)** | Replace the fixed budget with per-node memories of capacity C, filled by stochastic generation. Stored pairs decohere (visibility v·λ^age), and a request's gain depends on the age of the pair it receives through its own threshold v*(τ). | Spending now versus later becomes an age-dependent, request-specific assignment. A single scalar threshold cannot price pairs of different ages against requests with different fidelity needs. | Exact DP over (public state, age multiset), feasible for C ≤ 4 with an age cutoff at the point where every gain is zero. |
| **B. Hidden regime (BellRoute-H)** | Each episode draws a latent regime (for example "rich" or "lean" future demand) that changes the transition matrix. The current state has the same marginal under both regimes, and only the history is informative. | Memoryless pacing rules cannot adapt to the regime. Recurrent learners can, in principle. | Belief-state DP over (state, regime posterior) on the small chain. |
| C. Unknown state tables | Hide the state table, removing CE-DP's information advantage. | It does not challenge bid-price, which never used the table. | Not recommended. |

## Protocol (mirrors Stages 25–28)

1. **31a: design and headroom pilot (development data only).**
   - Implement A and B, their exact references, and strengthened equal-information rules:
     - an age-aware bid-price rule for A;
     - a regime-filtering bid-price rule for B;
     - a one-step rollout heuristic.
   - Register a headroom diagnostic on development catalogs.
   - **Gate:** keep a variant only if its best tuned equal-information rule stays at or below 95% of the optimum. If neither variant qualifies, stop and report the null: the BellRoute family is solved by simple rules. That result can be added to the TMLR paper.
2. **31b: fresh splits.**
   - Regenerate request types and catalogs with collision-free seed derivation (hash-based, with a test that rejects duplicated dynamics across splits). This fixes the Stage-29 split leak.
   - Hash-lock the test split before any learner runs.
3. **31c: development.**
   - PPO, recurrent PPO, and DQN, each with an equal and larger tuning budget than before: 6 configurations × 3 seeds, longer training, and a calibrated PPO value baseline.
   - The rules get the same validation budget.
   - Select the candidate learner and the strongest rule on validation only.
4. **31d: locked test.**
   - Train 12 fresh seeds of the candidate learner and evaluate once.
   - **Primary test:** candidate learner minus the best tuned equal-information rule, with a seed-level 95% bootstrap interval. The result is superior, inferior, or inconclusive, by a rule registered in advance.
   - **Secondary:** the learner as a fraction of the optimum, the learner versus greedy, and robustness to noise.

## Budget (to confirm with the user)

- **CPU only.** The estimated total is 6–8 active wall-hours (Stage 28 used 2.3 h for 24 models), with a cumulative cap of 10 h and at most 4 threads. No GPU and no shots.
- **The 31a pilot is cheap** (minutes, with no training) and decides whether the rest is worth running.

## Relation to the TMLR paper

The current manuscript stands on its own and can be submitted now. A positive Stage 31 result would make a separate follow-up or a revision. A null result from 31a could be added as one paragraph.

## Decisions needed from the user

1. Approve variant A, B, or both for the 31a pilot.
2. Approve the 10-hour CPU cap for 31b–31d. This does not apply if the 31a gate stops the stage.
3. Choose whether to submit the TMLR paper now or wait for Stage 31.
