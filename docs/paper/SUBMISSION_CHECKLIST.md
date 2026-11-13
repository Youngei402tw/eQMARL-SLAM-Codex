# QIP submission checklist

Status key: [x] done and verified in the repository; [ ] needs author action.

## Manuscript

- [x] Title page with title, author, affiliation, and e-mail.
- [x] Abstract (single paragraph) and keywords.
- [x] Sections: introduction, related work, problem definition (plain-language game, purpose of outputs, exact winning condition, worked example), formulation, learner and baseline definitions, fairness table, protocol, data splits, results (Study I, development and ablations, locked confirmatory study, transfer and scaling, noise and thresholds), audit and compute accounting, discussion, limitations, conclusion, and declarations.
- [x] Every result number comes from `generated/numbers.tex`, which is generated from the Stage 29 audit.
- [x] Every data figure has committed source CSVs in `figure_data/`, hash-checked against the audit bundle.
- [x] Compiles with no LaTeX errors, undefined citations or references, or overfull boxes.
- [x] PDF metadata (title, author, subject, keywords) has been set.
- [x] All pages have been inspected visually.
- [x] Null and failed results are reported: the 90% gate, Double DQN, PPO > DQN, Holm catalog contrasts, and Werner collapse.
- [x] The split-overlap finding is disclosed, with a post hoc analysis restricted to the fully fresh catalogs.
- [x] Terminology: the allocator is centralized over public information, and the endpoint games are distributed. There are no claims of a Bell violation, speedup, hardware evidence, or general advantage.

## Author actions before submission

- [ ] Confirm the author list, affiliation, and corresponding-author details. The affiliation was inferred from the e-mail domain.
- [ ] Transfer the manuscript to the Springer Nature LaTeX template (`sn-jnl.cls`, `sn-mathphys-num` style) if the journal requires it at first submission. The template was not available offline in the build environment. The content uses standard packages and should port directly.
- [ ] Check the journal's current policy on statements: data availability, code availability, AI-tool use, funding, competing interests, and author contributions. Edit the drafts in "Statements and declarations" as needed.
- [ ] Verify the bibliographic details of the two arXiv preprints (Haldar et al., arXiv:2303.00777; Li et al., arXiv:2412.06938), and update them if they have since been published.
- [ ] Optionally run a systematic literature search on entanglement allocation and routing with learned controllers, then revise the novelty statement.
- [ ] Optionally archive the tagged release on a DOI-minting service such as Zenodo, and cite the DOI in the data-availability statement.
- [ ] Review and sign `COVER_LETTER.md`.
- [ ] Suggest reviewers, if the submission system asks for them. None are proposed here.
