# TMLR submission checklist

**Target venue: *Transactions on Machine Learning Research* (TMLR).**
- TMLR accepts papers whose claims are correct and supported by evidence, and that some part of the community would find interesting. It does not require novelty or state-of-the-art results, so a well-documented negative result fits.
- Review is double-blind and submission is rolling.
- An accepted paper that receives a J2C, Featured, or Outstanding certification can be presented through the NeurIPS/ICLR/ICML journal-to-conference track.
- Check the current rules before submitting: <https://jmlr.org/tmlr/author-guide.html>.

## Files

| File | Purpose |
|---|---|
| `BELLROUTE_TMLR.pdf` | Anonymized submission PDF (TMLR style, 16 pages including appendix) |
| `BELLROUTE_TMLR.tex`, `references.bib` | Source |
| `BELLROUTE_TMLR-source.zip` | Self-contained source that compiles on its own (`make submission`) |
| `make_tmlr_assets.py` | Regenerates `generated/numbers30.tex`, `figure_data/*.csv`, and `figures/*.pdf` from the hash-checked Stage-28, Stage-30 and Stage-31 bundles |
| `figure_data/SHA256SUMS` | Hashes of every plotted data file |

To rebuild: `make PYTHON=/path/to/venv/python` (needs latexmk). Stage-28/29 numbers come from `../generated/numbers.tex`.

## Before you submit (author actions)

1. **Read the whole paper and agree with every claim.** You are responsible for it, and the AI-use statement says so.
2. **Do not submit the QIP manuscript at the same time.** It also states the request weights as w ∈ {1,2,3}, but the data use {1,2,3,4}; the TMLR version is corrected. TMLR forbids dual submission. The two papers share data, so choose one. This version reports the stronger-baseline result, and the QIP version predates it and overstates the learned claim. If you still want a QIP paper, it must be revised to include the Stage-30 result.
3. **Prepare anonymized supplementary material** (at most 100 MB, ZIP).
   - The repository is not anonymous: its name, git authors, `AGENTS.md`, `project_state.json`, and the manifests contain your identity and GitHub URL.
   - Either upload a scrubbed copy of `eqmarl_slam/`, `configs/`, `tests/`, `experiments/28`, `experiments/30`, `experiments/31`, and `scripts/`, or use an anonymizing mirror such as anonymous.4open.science.
   - Do not link to the GitHub repository during review.
4. **Check that the paper's claims match the archived outcomes**:
   - the registered Stage-28 decisions;
   - the Stage-30 decision `bid_price_superior`;
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
| "The learners were under-tuned." | §6.3 "Learner limitations" concedes this and narrows the claim to what was tested under equal opportunity. |
| "Is the quantum part necessary?" | Introduction and §3.1. The quantum part is presented as a source of exactly certified rewards and of a fidelity threshold, not as a quantum-advantage claim. |
| "Maybe a different variant would favour learning." | §6.5: two harder variants were registered and tested. The hidden-regime study stopped at its preregistered futility rule, and its fresh test split was never opened. |
| "Split leakage." | §7. All results are reported on the fully fresh catalogs as well. |
