# Stage 02 — Controlled known-pose exploration

Status: running. Dependency completion: `6b49e7010d9c0667acb41854b1d008f303ea0355`.
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
