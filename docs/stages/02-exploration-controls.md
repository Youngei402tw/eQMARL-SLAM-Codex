# Stage 02 — Controlled known-pose exploration

Status: passed. Dependency completion: `6b49e7010d9c0667acb41854b1d008f303ea0355`.
Branch `stage/02-exploration-controls` was created from that exact commit.

## Question and falsifier

Can the grid task, planner and evaluator measure exploration without leaking
teammate information or confusing mapped cells with physical sensing? The
smallest falsifier is a five-cell corridor whose fixed physical free-cell
coverage fails to change from 4/5 to 1 after the registered sensing moves, or
whose local actor input changes when only a teammate's private map changes.

## Registered contracts and gates

This is a known-pose grid exploration diagnostic. There is no uncertain pose
estimator, learned controller, or SLAM-performance claim. Coordinates are row,
column; primitive actions are wait, north, east, south and west. Sensors are
cardinal rays with first-wall occlusion. The horizon is intrinsic and remaining
budget is observed. Simultaneous collision handling preserves legal distinct
robot positions. Rewards, horizon, sensor and metric definitions are identical
across controls.

Local-only random, reactive and frontier controllers receive only their own map,
pose, visit history and legal-action mask. Shared frontier assignment receives a
declared transmitted shared map and poses; bytes/messages are charged and its
information advantage is explicit. It is not an equal-information control.

Physical-free coverage uses a fixed truth-defined reachable-free domain;
estimated coverage is computed separately from mapper state. Map accuracy uses
the whole fixed grid with unknown cells counted incorrect. A mapper-corruption
fixture must change estimated coverage without changing physical coverage.

Required fixtures cover corridor coverage, occlusion, unreachable/current-cell
targets, bounds/walls/same-target/swaps/occupied-cell collisions, information
isolation, shared payload accounting, masks, intrinsic horizon, and explicit
sampled/greedy modes. Frontier controls must complete the corridor and improve
coverage on the open map. Failed or stalled development episodes remain in the
raw results; no learned-method ranking is made.

The registered development matrix uses fixed open, bottleneck and dead-end maps,
three development seeds, four controls, and at most 80 primitive steps per
episode. Diagnostic family cap is 3,600 seconds and cumulative stage cap is
86,400 seconds. One CPU thread; no quantum calls, physical shots, or policy
training. Code/configs are published and verified before executing the matrix.

## Evidence

Run definitions, outcomes and artifact identities will be recorded in
`experiments/02/index.json`. All raw per-step and episode records, exact maps,
resolved settings and fixture evidence are permanent Git artifacts.

## Observed results and limitation diagnosis

All 14 registered correctness/progress gates passed and the 36-episode matrix completed. There were 1,709 joint primitive transitions including seven scripted fixture transitions, 3,418 robot decisions, no optimization updates, 3.042s recorded experiment wall, 0.000845 CPU core-hours, no GPU use and no quantum calls. The separate conservative setup envelope is charged in project_state.json.

The shared-map frontier controller completed all nine episodes across all three maps. Local-only frontier completed bottleneck and dead-end maps but stalled at 27/35 physical free cells on the open map. Inspection of the retained trace shows repeated east/west collision attempts with no planner progress (152 blocked robot moves by step80 for each deterministic seed repeat). This is a stateless coordination limitation under the registered local-only sensing contract, not an unreachable-target or physical-coverage accounting error. Reactive controls also stall; random controls fail some episodes. Every episode remains in the raw data. No controller was changed after observing these outcomes.

The shared controller explicitly receives exchanged maps and poses and incurs payload costs, so its completion rate is not an equal-information policy improvement claim. Deterministic greedy repeats are not independent trained models. Physical and estimated coverage agree in noiseless known-pose episodes; the separate corruption fixture proves their accounting is independent. Full-grid mapping accuracy remains below100% in some physically covered maps because unreachable/occluded walls remain unknown; coverage is not relabeled mapping accuracy.

All required maps, 1,785 raw records, trajectories, diagnostics and source identities were retrieved from GitHub and validated. The full source suite passed76 tests before execution. Stage03 can proceed to genuine uncertain poses and unknown landmarks; this stage makes no SLAM, learning or quantum claim.
