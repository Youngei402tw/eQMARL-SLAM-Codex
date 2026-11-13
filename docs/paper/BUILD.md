# Building the QIP manuscript

The manuscript `QIP_ROBUST_ENTANGLEMENT_ROUTING.tex` takes every result number and every data figure from the Stage 29 independent audit bundle `experiments/29/29-qip-release-audit_attempt-01/`. That bundle is itself re-derived from the raw Stage 23, 26, 27, and 28 records.

## 1. Environment

- Python 3.12 with `requirements.lock`, plus `matplotlib==3.10.7` (figure generation only).
- TeX Live 2023 or later with `latexmk`, `pdflatex`, `bibtex`, `tikz`, `natbib`, `cleveref`, `tabularx`, `booktabs`, and `lmodern`.

## 2. Reproduce the audit (optional; the bundle is committed)

The audit performs no allocator decisions or training, and it runs in about ten seconds on one CPU thread:

```bash
python -m eqmarl_slam.cli verify --run experiments/29/29-qip-release-audit_attempt-01
```

A fresh re-execution needs a new attempt ID and a published source receipt, as described in `docs/GITHUB_WORKFLOW.md`.

## 3. Regenerate numbers and figures, then compile

```bash
cd docs/paper
make qip PYTHON=python
```

`make qip` runs the following steps:

1. `make_numbers.py` writes `generated/numbers.tex`, which contains about 200 macros. The prose and tables use these macros for results and never hard-code them.
2. `make_figures.py` checks the SHA-256 of each figure-source CSV against `paper_tables.json` in the audit bundle. It then copies the CSVs to `figure_data/` and writes the vector PDFs in `figures/`.
3. `latexmk` compiles `QIP_ROBUST_ENTANGLEMENT_ROUTING.pdf`.

## 4. Figure sources

| Figure | Content | Source data (`figure_data/`) |
|---|---|---|
| 1 | Round of budgeted routing (TikZ diagram) | none (schematic) |
| 2 | Example request graph and XOR rule (TikZ) | `configs/data/qip_v1/test-topology-a6-ring-catalog.json`, `experiments/28/.../test_certificates.json` |
| 3 | Development ablations and learner comparison | `fig_ablation_stage26.csv`, `fig_stage26_methods.csv` |
| 4 | Sealed-catalog surplus by method | `fig_main_by_catalog.csv`, `fig_seed_catalog.csv` |
| 5 | Primary intervals | `fig_primary_intervals.csv`, `fig_seed_contrasts.csv` |
| 6 | Scaling across agent counts | `fig_scaling.csv` |
| 7 | Heralded loss, Werner noise, thresholds | `fig_noise_stage27.csv`, `fig_noise_stage28.csv`, `fig_visibility_thresholds.csv`, `fig_positive_fraction_test.csv` |

`fig_training_curves_summary.csv` holds the per-seed training diagnostics: final 100-episode return and loss, wall time, and updates.

## 5. Checks run before release

- The `latexmk` log has no errors, no undefined citations or references, and no overfull boxes.
- `pdfinfo` reports the title, author, subject, and keywords.
- `pdftotext` output was compared against `generated/numbers.tex`.
- Every page was rendered and inspected visually.
