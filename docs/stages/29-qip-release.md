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
