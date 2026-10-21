# Stage 09 — Reproducible release and handoff

Status: passed. The clean-runtime diagnostic, compact smoke replay, raw-analysis regeneration, all-stage evidence audit, final report, and continuation guide are complete.

## Question and implementation

The stage tested whether a fresh environment built only from the committed lockfile could verify every frozen Stage 08 checkpoint, replay saved policies, and regenerate the locked analysis without retraining or mutating evidence.

The environment used Python 3.12.14 with no inherited system site packages. `pip check` and all 112 unit tests passed against the exact lockfile. The registered diagnostic loaded all 24 Stage 08 model/optimizer checkpoint payloads and evaluated finite actor probabilities. It replayed the first validation scenario for seed 8801 from each confirmatory method using actor-only evaluation, then regenerated the locked analysis from archived raw metrics. The standalone smoke path replayed the centralized-classical seed-8801 actor and is recorded in `analysis/09/reproduce-smoke.json`.

## Outcomes

Attempt 01 completed all read-only diagnostic checks but failed its final archive validation because the resolved model metadata omitted the schema-required `encoder` and `parameter_counts` fields. Its 28 raw records, regenerated analysis, diagnostics, 18 primitives, and 5.387 wall-seconds are retained in a valid failed bundle.

Attempt 02 corrected only the archive metadata contract and reused the frozen inputs. It passed every gate in 5.547 wall-seconds: all 24 registered checkpoint hashes matched and loaded, all three validation replays matched every registered metric below `1e-8`, and regenerated analysis was byte-identical at SHA-256 `800c7d1e03bc828f4d8430f79515402ebbd4161a614f6e82aa566bcaa5648d56`. It used 18 primitives, 12 decisions, zero optimizer updates, zero circuit evaluations, and zero shots.

Fresh remote retrieval at results commit `d9e130c99338200ae144c558735226ac3a6dc24d` validated both attempts and every inventoried file. The receipt is `docs/stages/09-publication.json`.

## Full evidence audit

The raw-only audit found 84 bundles across Stages 01–09 and no planned run without an outcome. Four failed attempts remain in the ledger: one Stage 03 input-format failure, two Stage 04 preflight/archive failures, and Stage 09 attempt 01. No inventoried artifact exceeds 10 MiB.

Recorded experiment segments total 348,262 primitives, 247,472 decisions, 19,241 updates, 242,371 analytic/simulated circuit evaluations, 1,378,304 simulated shots, 7,171.420 wall-seconds, and 1.99185 CPU-core-hours. No GPU time or physical hardware shots were used. Setup/development active-wall budgets remain separate in `project_state.json`; their CPU and circuit costs were not fully instrumented.

The stage consumed 718.372 active wall-seconds including clean-environment setup, tests, both attempts, publication, and audit work, below the 86,400-second cap. Both attempts together account for 10.934 measured experiment wall-seconds and 0.00304 CPU-core-hours.

## Gate decision

Passed. A fresh session can identify the completed state, install the lockfile, run `scripts/reproduce_smoke.py`, regenerate Stage 08 analysis, and rebuild the release inventory. All required evidence is remotely retrievable with verified hashes. The final scientific conclusion remains the bounded-budget null result described in `docs/FINAL_REPORT.md`; the release makes no convergence, equivalence, scaling, hardware, or general quantum-advantage claim.
