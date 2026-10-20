# Experiment archives

No training or evaluation runs have been executed in Stage 00. This directory intentionally contains no synthetic run results.

At subsequent stages, create `experiments/<stage>/<run_id>/` according to `docs/EXPERIMENT_PROTOCOL.md` and `docs/GITHUB_WORKFLOW.md`. Publish raw metrics, complete configs, seed manifests, diagnostics, failures, and durable checkpoint references. Maintain a per-stage run index including every planned/started run and its outcome.

The JSON schema under `schemas/` defines required manifest fields. Stage 01 implements runtime validation and the archive CLI before reportable training begins.
