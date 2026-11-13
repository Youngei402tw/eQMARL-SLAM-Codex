# Stage 31 — Can learning beat strong rules when pacing is not enough? (proposal)

**Status: 31a in progress.** The user said "continue", which authorizes only the cheap headroom pilot (31a): development and validation data, no training, no test data. Learner training (31b–31d) still needs explicit approval.

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

## 31a registration

Run `31-headroom-pilot_attempt-01` is registered as planned (`configs/stage31/31-headroom-pilot_attempt-01.yaml`).

**Fixed settings.**
- Variant A: capacity 3, generation probability 0.25 per round, storage decay 0.985 per round.
- Variant B: tilt beta = 1.

**Disclosure.** These settings were chosen in an unregistered exploratory sweep on the development catalogs. In that sweep, rules tuned separately for each catalog reached 94.5–96% of the optimum for A and 95.6–96.4% of the belief-state optimum for B.

**Design.** The registered run tunes each rule with a single shared parameter set on the development catalogs and then scores it on the four held-out validation catalogs. The gate is 95%.

## 31a result

Run `31-headroom-pilot_attempt-01` ran from source `1e12f30cd092b7bae8b6d74fdc33be567216aab8`. Its status is **completed**, and its gates passed. Registered decision: **`proceed_with:A_generate_and_store,B_hidden_regime`**.

| Variant | Greedy / reference | Best dev-tuned rule / reference (validation) | Worst catalog | Selected rule |
|---|---|---|---|---|
| A, generate-and-store | 93.6% | **94.6%** | 93.8% | alpha=2.0, omega=40 |
| B, hidden regime | 59.2% | **94.8%** | 86.6% | alpha=0.5, omega=20 |

In variant B, knowing the regime outright is worth only 3.6% over the belief-state optimum.

**Interpretation.** Both variants pass the registered 95% gate, but by small margins. The best simple rule leaves only about 5% of the reference surplus for learning to capture. In v1 the learned allocators finished about 11 points *below* bid-price. Beating a rule that is already within about 5% of optimal would therefore require a large improvement in learner quality. These are thin margins, and a negative 31d outcome is the more likely one.

**Resources.** 57 wall-seconds, 805,000 decisions, no training, no test data.

**Next.** Stages 31b–31d (fresh collision-free splits, learner development, one locked test) need explicit user approval for a CPU budget of up to 10 h.

## 31b–31d registration (user approved the full study, cap 10 CPU wall-hours)

**Variant.** Stage 31 continues with variant **B (hidden regime) only**. It uses the existing learners' action space; variant A would need a new action encoder and is not pursued.

**31b: fresh test split.** `scripts/generate_qip_v2_test.py` writes `configs/data/qip_v2/`:
- 16 fresh request types and four sealed catalogs: 4-agent complete, 5-agent ring, 8-agent sparse, and 6-agent path with H = 24;
- seeds derived from SHA-256 of unique labels;
- generation fails on any reuse of v1 dynamics or request contents.

The split lock is committed and hashed in the development config. Its contents have not been inspected.

**31c: development** (`31-regime-development_attempt-01`).
- PPO, recurrent PPO, and DQN are tuned equally: 6 configurations × 3 seeds × 2,000 episodes on the development regime maps. PPO now uses a calibrated value baseline.
- Each learner is scored on the validation catalogs (300 sequences each).
- The bid-price rule is tuned on validation over 35 configurations.
- **Futility rule:** stop before the locked test if the best learner's validation mean is below the best rule's mean minus 5% of the belief-state DP mean.

**31d: locked test** (to be registered after development, with the freeze hash).
- 12 fresh seeds of the candidate learner, 4,000 episodes each, trained on development and validation.
- One evaluation on qip_v2, 500 sequences per catalog.
- **Primary:** candidate minus frozen bid-price, seed bootstrap. The result is learned superior, rule superior, or inconclusive.
