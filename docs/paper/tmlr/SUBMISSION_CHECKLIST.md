# TMLR submission checklist

**Target venue: *Transactions on Machine Learning Research* (TMLR).**
- TMLR accepts papers whose claims are correct and supported by evidence, and that some part of the community would find interesting. It does not require novelty or state-of-the-art results, so a well-documented negative result fits.
- Review is double-blind and submission is rolling.
- An accepted paper that receives a J2C, Featured, or Outstanding certification can be presented through the NeurIPS/ICLR/ICML journal-to-conference track.
- Check the current rules before submitting: <https://jmlr.org/tmlr/author-guide.html>.

## Files

| File | Purpose |
|---|---|
| `BELLROUTE_TMLR.pdf` | Anonymized submission PDF (TMLR style, 21 pages including appendix) |
| `BELLROUTE_TMLR.tex`, `references.bib` | Source |
| `BELLROUTE_TMLR-source.zip` | Self-contained source that compiles on its own (`make submission`) |
| `make_tmlr_assets.py` | Regenerates `generated/numbers30.tex`, `generated/main_table.tex`, `generated/xor_example.tex`, `generated/catalog_table.tex`, `generated/strength_table.tex`, `generated/transition_table.tex`, `figure_data/*.csv`, and `figures/*.pdf` (including `fig_network.pdf`) from the hash-checked catalogs, certificates, and Stage-28 and 30–33 bundles |
| `figure_data/SHA256SUMS` | Hashes of every plotted data file |

To rebuild: `make PYTHON=/path/to/venv/python` (needs latexmk). Stage-28/29 numbers come from `../generated/numbers.tex`.

## Before you submit (author actions)

1. **Read the whole paper and agree with every claim.** You are responsible for it, and the AI-use statement says so.
2. **Do not submit the QIP manuscript at the same time.** It also states the request weights as w ∈ {1,2,3}, but the data use {1,2,3,4}; the TMLR version is corrected. TMLR forbids dual submission. The two papers share data, so choose one. This version reports the stronger-baseline result, and the QIP version predates it and overstates the learned claim. If you still want a QIP paper, it must be revised to include the Stage-30 result.
3. **Prepare anonymized supplementary material** (at most 100 MB, ZIP).
   - The repository is not anonymous: its name, git authors, `AGENTS.md`, `project_state.json`, and the manifests contain your identity and GitHub URL.
   - Either upload a scrubbed copy of `eqmarl_slam/`, `configs/`, `tests/`, `experiments/28`, `experiments/30`, `experiments/31`, `experiments/32`, `experiments/33`, and `scripts/`, or use an anonymizing mirror such as anonymous.4open.science.
   - Do not link to the GitHub repository during review.
4. **Check that the paper's claims match the archived outcomes**:
   - the registered Stage-28 decisions;
   - the Stage-30 decision `bid_price_superior`;
   - the Stage-31 futility stop and the Stage-32 decision `gap_persists`;
   - the Stage-33 decisions (H1 supported, H2 not supported for successor 0.8, H3 supported);
   - the audit's material finding on the split leak.
   All numbers are macros generated from the bundles; do not hand-edit them.
5. **On OpenReview**:
   - Fill in the submission form, including the TMLR LLM-use questions if present.
   - Suggest action editors with RL-evaluation or operations-research expertise.
   - Choose a subject area such as "Reinforcement learning" or "Benchmarks/evaluation".
6. **After acceptance**:
   - Switch to `\usepackage[accepted]{tmlr}`.
   - Add the author block, month, year, and OpenReview link.
   - Replace the supplementary-material sentence with the public repository URL.

## Points reviewers are likely to raise, and where the paper answers them

| Likely objection | Where the paper answers it |
|---|---|
| "The baselines were added after the test set was opened." | §5 and Limitations. The comparators were registered before running, tuned only on validation, and the selection was written to disk before any sealed file was read. They never informed a locked decision. The result also goes *against* the authors' learned method. |
| "The learners were under-tuned." | §6.3 and Appendix D. A registered learner-strength check (Stage 32; wider grid, about 3× longer training, calibrated PPO value baseline, 6 seeds) gave `gap_persists`: PPO reached 88.2% of the optimum against 96.7% for the rule, CI of the difference [−0.037, −0.028]. The remaining caveat is that only MLP learners were tested. |
| "Is the quantum part necessary?" | Introduction and §3.1. The quantum part is presented as a source of exactly certified rewards and of a fidelity threshold, not as a quantum-advantage claim. |
| "CE-DP only works because the generator's dynamics happen to match its default guess." | §6.3 "How far the dynamics result extends" and Appendix E (Stage 33). Pacing holds under all 7 rebuilt transition matrices (94.9–97.8%). The default guess holds at ≥ 98.6% except under strong successor structure (96.5%), where 10–100 logged episodes recover 98.9–99.9%. |
| "Maybe a different variant would favour learning." | §6.5: two harder variants were registered and tested. The hidden-regime study stopped at its preregistered futility rule, and its fresh test split was never opened. |
| "Split leakage." | §7. All results are reported on the fully fresh catalogs as well. |

## Overleaf sync

The manuscript is mirrored to a separate private repository, `bellroute-tmlr-paper`, which is linked to Overleaf through GitHub sync. There the paper is `main.tex`.

- **Export from here:** run `make overleaf DEST=/path/to/bellroute-tmlr-paper` and commit in that checkout. This replaces only the paper files.
- **Edits made in Overleaf:** push them to GitHub from Overleaf, pull them in the mirror checkout, and port the prose changes back into `BELLROUTE_TMLR.tex` here.
- **Generated files:** never hand-edit `generated/` or `figures/` in the mirror.
