# Stage 29 — QIP paper and independent release audit

## Preflight

**Question.** Do independent reconstructions of the locked Stage 28 study and the earlier QIP-extension stages, built from raw rows, certificates, checkpoints, and git history, reproduce every registered decision, interval, threshold, and resource count? What is the strongest claim that a *Quantum Information Processing* manuscript can defend?

**Falsifiers.** The release fails if any of the following holds:
- a reproduced Stage 28 mean, interval, Wilcoxon p-value, Holm adjustment, secondary-condition mean, or visibility threshold differs from the recorded value by more than `1e-9`;
- any registered decision changes;
- a checkpoint hash, identity, or frozen configuration does not match;
- any learned row violates the budget identity;
- request-type splits overlap;
- a sealed file changed after the Stage 25 lock;
- resources do not reconcile or exceed a cap;
- a null or failed claim is omitted from the manuscript.

**Design.** `eqmarl_slam/diagnostics29.py` is written independently of the Stage 28 analysis. It does not import `diagnostics28`, the routing environment, or the learners. It uses its own exact classical enumeration, analytic Bell evaluation at the recorded angles, its own XOR SDP, and its own bootstrap, exact signed-rank, and Holm implementations. It checks for overlap in public dynamics across catalogs, and it reports a clearly labelled post hoc descriptive sensitivity on the catalogs whose dynamics are fresh. Its outputs include the manuscript tables (`paper_tables.json`) and the hashed source CSV of every data figure.

**Budget.** At most one active wall-hour, 0 allocator decisions, 0 training, at most two CPU threads, and no shots. The registered run is `29-qip-release-audit_attempt-01`.

**Deliverables.**
- The audit bundle.
- `docs/paper/QIP_ROBUST_ENTANGLEMENT_ROUTING.{tex,bib,pdf}`.
- Generated numbers and figures, with their source data.
- Build instructions, a submission checklist, a cover-letter draft, and a contributions-and-limitations summary.

**Gate.** All audit gates pass, the manuscript compiles cleanly with claims that match the registered outcomes, and every artifact is remotely retrievable on `stage/29-qip-release`. The repository workflow must also pass.

## Audit result

Run `29-qip-release-audit_attempt-01` ran from source `865932a368e4a280165c583d59b2550c44769553`. Its status is **completed**. 18 of 18 audit gates passed: `all_bundles_valid`, `freeze_hash_matches`, `bundle_source_is_frozen_commit`, `bundle_status_completed`, `bundle_config_hash_matches_committed_config`, `raw_rows_complete_and_paired`, `registered_analysis_reproduced`, `decisions_stable_under_alternate_bootstrap_stream`, `certificates_independently_verified`, `visibility_thresholds_reproduced`, `checkpoints_hash_identity_and_config`, `exactly_one_final_checkpoint_per_model`, `budget_and_matching_safety`, `request_type_splits_disjoint`, `test_lock_compliance`, `resource_accounting_reconciled`, `within_registered_caps`, `earlier_stage_numbers_reproduced`.

24 of 24 registered Stage 28 quantities were reproduced within 1e-9. These are the method means, the three primary intervals, the four exact Wilcoxon p-values with their Holm adjustments, the secondary-condition means, and the visibility thresholds. The registered decisions are unchanged: {"ppo_superior_to_dqn": false, "ppo_superior_to_greedy": true, "ppo_dqn_equivalent_within_5pct_online_reference": true}.

### Findings

- **material** `test_public_dynamics_overlap`: Sealed test catalogs qip-test-horizon-a4-path, qip-test-id-a4-complete reuse the public Markov dynamics (transition matrix, active-edge pattern, urgencies) of training catalogs; only their request types are fresh. Fully fresh test catalogs: qip-test-topology-a6-ring, qip-test-scale-a8-sparse.
- **material** `validation_public_dynamics_overlap`: A validation catalog reuses a development catalog's public dynamics; Stage 26-27 validation was not fully independent for that catalog.
- **disclosure** `runtime_lock_drift`: Stages 26-28 ran with torch 2.15.0+cpu and numpy 2.5.4 while requirements.lock pins torch 2.14.1+cpu and numpy 2.4.6; the manifests record the actual versions.
- **disclosure** `ppo_value_baseline_scale`: The PPO value head is fit to standardized returns, while the advantage subtracts its raw output from unstandardized returns; it acts as a learned state-dependent offset rather than a calibrated baseline. Retained as implemented and frozen.
- **disclosure** `updates_counter_mixes_sources`: The Stage 28 'updates' counter sums allocator optimizer updates and request-certification optimizer updates; the audit separates them.
- **disclosure** `secondary_gate_constant`: Stage 28 sets registered_secondary_conditions_complete=True unconditionally; the audit independently verifies 19,200 secondary rows with exact pairing.
- **disclosure** `matching_not_logged`: Per-step matchings are not stored in raw rows; feasibility rests on the evaluator's raising check and the empty violation list.

### Resources

The audit used 3.65 wall-seconds and 0.00099 CPU core-hours. It made 0 allocator decisions, ran no training, made 16 analytic Bell evaluations, solved 16 SDPs, and sampled 0 shots.

### Manuscript

The manuscript is `docs/paper/QIP_ROBUST_ENTANGLEMENT_ROUTING.{tex,bib,pdf}`. All result numbers come from `docs/paper/generated/numbers.tex`, and all data figures come from `docs/paper/figure_data/*.csv` (hash-checked).
The supporting files are `docs/paper/BUILD.md`, `SUBMISSION_CHECKLIST.md`, `COVER_LETTER.md`, and `CONTRIBUTIONS_AND_LIMITATIONS.md`.
