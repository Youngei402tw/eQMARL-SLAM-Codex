# eQMARL-SLAM-Codex

A staged research project to test whether an entangled quantum critic improves cooperative active SLAM under controlled information, model, and compute budgets.

**Stages 00–09 are complete with verified evidence.** The locked eight-seed confirmatory study found an eQMARL-minus-matched-no-entanglement physical-coverage difference of 0.00431 with a 95% paired-seed interval of [-0.01458, 0.02615]. This does not establish superiority or equivalence. A separately authorized prospective [mechanism-focused extension](docs/MECHANISM_STUDY_PLAN.md) begins at Stage 10 with fresh data and a complementarity-by-entanglement interaction hypothesis. See [the final core report](docs/FINAL_REPORT.md) and [project state](project_state.json).

## Start here

- [Codex instructions](AGENTS.md): what to read, check, implement, archive, and report at every stage.
- [Execution plan](docs/EXECUTION_PLAN.md): ten stages with dependencies, experiments, exit gates, and failure routes.
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md): fair comparisons, data splits, statistics, resource limits, and run records.
- [Framework decision](docs/adr/0001-pytorch-stack.md): PyTorch, PennyLane, Gymnasium, and a 2D estimator adapter.
- [Reference audit](docs/REFERENCE_AUDIT.md): what is retained from eQMARL and what changes from the existing SLAM implementations.
- [GitHub workflow](docs/GITHUB_WORKFLOW.md): permanent stage branches and verified experiment publication.
- [Research roadmap](docs/RESEARCH_ROADMAP.md): feasibility and contribution ranking, including optional ROS and 3D extensions.
- [Stage registry](stages.json), [Stage 08 confirmatory report](docs/stages/08-confirmatory.md), and [Stage 09 release report](docs/stages/09-release.md).

## Technical direction

Use PyTorch for the common classical encoder, recurrent actor, critic interfaces, and training loop. Use PennyLane QNodes with its Torch interface for the quantum critic. Retain the original TensorFlow/Cirq implementation as a pinned reference in a separate environment; first establish circuit and gradient parity, then introduce corrections under separate protocol identifiers.

Start with a small grid exploration diagnostic, followed by genuine 2D landmark SLAM with noisy body-frame odometry, range-bearing observations, and an estimator whose uncertainty changes on informative revisits. A common high-level actor selects frontiers, revisits, or rendezvous; a shared planner handles motion. This isolates the critic experiment from low-level navigation difficulty.

The initial quantum component is used during training. Actors execute classically from local histories and explicitly allowed messages. Any shared map, relative observation, factor exchange, or quantum resource must have a declared information and cost contract.

A positive, null, or negative quantum comparison can all complete the project. Correctness, reproducibility, and a useful scientific answer determine stage completion.

## Stage branches

| Stage | Branch | Purpose |
|---|---|---|
| 00 | `stage/00-plan` | Design, protocols, reference pins, archive policy |
| 01 | `stage/01-reference-port` | Dependency lock and faithful PyTorch/PennyLane port |
| 02 | `stage/02-exploration-controls` | Honest grid observations, metrics, and baseline diagnostics |
| 03 | `stage/03-slam-beliefs` | Genuine 2D estimator and loop-closure tests |
| 04 | `stage/04-classical-controller` | Common recurrent controller and strong classical baselines |
| 05 | `stage/05-critic-ablation` | Matched classical, no-entanglement, and eQMARL critics |
| 06 | `stage/06-interaction-model` | Proposed pairwise interaction critic and matched classical control |
| 07 | `stage/07-shot-noise` | Finite-shot, noise, and resource analysis |
| 08 | `stage/08-confirmatory` | Frozen protocol and held-out multi-seed evaluation |
| 09 | `stage/09-release` | Reproduction bundle, final report, and claim audit |
| 10 | `stage/10-mechanism-plan` | Prospective mechanism-study design and gates |
| 11 | `stage/11-mechanism-task` | Controlled complementarity task and diagnostics |
| 12 | `stage/12-mechanism-development` | Equal-budget tuning, pilots, and freeze |
| 13 | `stage/13-mechanism-confirmatory` | Locked interaction phase diagram |
| 14 | `stage/14-mechanism-release` | Reproduction and paper-ready contribution audit |

The completed Stage 00–09 branches were created from their recorded dependencies and are retained. Stage 10 is the active prospective planning branch; later extension branches remain planned until their stages begin. `main` is the stable integrated core release.

## What is runnable today

The Python 3.11/3.12 CPU stack, circuit port, SLAM task, recurrent MAPPO learner, matched critic families, checkpoint tools, numerical diagnostics, and archive verifier are implemented. Install `requirements.lock` with the official CPU Torch extra index, or use `requirements-classical.lock` for a classical-only environment. Legacy reference requirements are isolated under `configs/stage01/legacy/`.

```bash
python scripts/validate_plan.py
python -m pytest -q
python scripts/reproduce_smoke.py --root .
python -m eqmarl_slam.release_audit --root . --output /tmp/eqmarl-release-audit
```

The smoke and audit commands are read-only. New reportable execution requires a preregistered config, a clean source tree, and a verified published-source receipt. Existing run directories are immutable; reproduction that creates a new bundle uses a separately registered attempt.

## Resume in Codex

Open this repository in Codex and use:

> Read AGENTS.md, project_state.json, stages.json, and docs/EXECUTION_PLAN.md. Resume the earliest authorized unfinished stage. Validate its dependency commit, protocol, budget, and publication route before execution. Work on its stage branch, preserve all run outcomes, publish and verify every experiment bundle, and update its stage report. Do not start confirmatory tests until the protocol is frozen. Report completed work, evidence, remaining limits, and the next stage.

The recorded plan is complete. The stage reports and immutable commits are the continuation record; no training job remains active.

## References and attribution

The reference is [eQMARL: Entangled Quantum Multi-Agent Reinforcement Learning for Distributed Cooperation over Quantum Channels](https://arxiv.org/abs/2405.17486), with the authors' [official repository](https://github.com/news-vt/eqmarl). The existing user repositories are [eQMARL-SLAM](https://github.com/Youngei402tw/eQMARL-SLAM) and [eQMARL-SLAM-NEW](https://github.com/Youngei402tw/eQMARL-SLAM-NEW), including its active-SLAM branch. Exact audited commits are in [references.json](references.json).

Stage 01 includes attributed source extracts and separately named faithful and corrected implementation paths. Preserve upstream notices when porting or adapting code; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
