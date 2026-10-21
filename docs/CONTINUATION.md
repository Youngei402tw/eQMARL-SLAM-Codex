# Continuation guide

The authorized Stages 00–09 are complete. Start by reading `AGENTS.md`, `project_state.json`, `stages.json`, `docs/FINAL_REPORT.md`, and the active branch/commit ledger. Do not move or delete the retained stage branches.

Create a clean environment with Python 3.11 or 3.12 and the official CPU PyTorch index:

```bash
python -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

Run the read-only frozen-policy smoke path and rebuild the final raw inventory:

```bash
.venv/bin/python scripts/reproduce_smoke.py --root .
.venv/bin/python -m eqmarl_slam.final_analysis --root . --output /tmp/eqmarl-analysis
cmp analysis/08/confirmatory-analysis.json /tmp/eqmarl-analysis/confirmatory-analysis.json
.venv/bin/python -m eqmarl_slam.release_audit --root . --output /tmp/eqmarl-release-audit
```

These commands load archived actors and read raw records. They do not retrain or modify experiment bundles. A new scientific hypothesis must use a new protocol revision, branch, development data, and reserved test set. Reasonable next projects include longer-budget convergence studies, more agents with explicit scaling costs, ambiguous landmark association, occupancy/3D mapping, or physical-hardware feasibility. None is part of the completed claim.

For interrupted future training, resume only at the documented new-episode checkpoint boundary or implement and validate a separate exact-resume protocol. Never replace an existing run directory; register a new attempt and retain failures.
