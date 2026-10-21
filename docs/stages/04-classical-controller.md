# Stage 04 — Common recurrent belief controller

Status: passed. Dependency: `b0476fe8bb632389d62319003ef1e1c3c9312513`.
Stage branch created from that exact Stage03 completion; earlier branches retained.

The question is whether a common recurrent macro controller can learn in a
tractable uncertain-pose/unknown-landmark task. The smallest falsifier is an
actor input changed by hidden simulator truth while its factor posterior is
held fixed, or incorrect primitive-duration return/terminal arithmetic.

The versioned task uses a known empty4m square, unknown landmark coordinates,
body-frame dynamics/sensors and the validated full-history smoother. Actors
receive pose/uncertainty and exploration/revisit/rendezvous candidates derived
from their graph, including explicitly broadcast raw factors. All factor
payloads are charged. Simulator truth generates sensors, shared rewards and
evaluation only. The selected settings and fixed train/development/validation
manifests are in configs; final test scenarios are reserved and unused.

First publish a bounded diagnostic source and run truth isolation, macro
discount checks, four fixed heuristics, a synthetic discrimination learning
check, and a16-step recurrent-PPO smoke path. The task must be tractable for a
competent heuristic. Benchmark throughput before preregistering pilots. The
GRU has independent agent memory, resets at episode boundaries, and uses full
episode backpropagation. Corrected GAE bootstraps external cutoffs and uses
actual macro durations. Target normalization is fixed scale10 for every
subsequent critic. Checkpoints retain model/optimizer/RNG state and explicitly
support new-episode boundaries; no exact mid-episode continuation is claimed.

Diagnostic cap1h; pilot per-run cap2h; cumulative stage cap24h, summed across
attempts and sessions. Existing CPU only, one job/thread at a time. Null and
failed runs are retained. No quantum comparison can advance past a broken
classical learning path.

The first v1preflight encountered an ill-conditioned bearing marginal after a revisit placed an estimated landmark0.0003m from the robot. No training ran. Numerical replay is retained. Task v2 declares a0.25m near-field detection limit (true and measured range), counts discarded events, and uses a0.5m revisit standoff. Physical/estimated coverage both use the corresponding annulus. This sensor/candidate change is explicitly versioned before pilots; v1results are not pooled withv2. A resource reconciliation retains the original failed bundle while correcting three uncounted partial-macro primitives and two decisions in the external stage ledger.

The passing v2preflight took3.18s including100 synthetic updates,8 heuristic episodes and16 task primitives with2 PPO updates. All nine gates passed; competent frontier/uncertainty controls completed both development maps in6 primitives. Based on that throughput, preregistered pilots use3 seeds,2000 training primitives each,30 greedy validation episodes, and a2h run cap (7000s internal training/evaluation watchdog). This is a bounded feasibility budget within the20k pilot maximum. Final test scenarios remain unused. All pilot sources/configs are published together and executed from the same clean immutable commit with artifacts outside that checkout, preserving source integrity across sequential jobs.

## Outcomes and limits

The corrected v2preflight passed all9 gates. Its100 synthetic discrimination updates reached100% accuracy, and the16-task-step PPO smoke produced finite connected gradients and a real checkpoint. The original v1native bearing failure and v2NumPy-boolean archive failure are retained, with original partial outputs, checkpoint, and explicit metadata repair. Only v2results enter subsequent comparisons.

All3 preregistered2000-step pilots completed with finite loss, actor/critic gradients and noncollapsed entropy. Greedy validation across30 maps per seed produced physical-coverage means0.82037,0.71389,0.91111 and full-success counts0,0,30. This is substantial training-seed variation. The first two seeds are neither dropped nor called successful. Value fitting and the successful third seed show a working learning path; the study does not claim robust mastery or superiority to the competent heuristics. Later comparisons must retain this limitation and tune each critic with equal declared opportunity.

Every pilot uses the same actor architecture, data catalogs, macro controller, duration-aware PPO and return scale10. They were executed sequentially from one clean immutable source commit e4a49ac, with outputs outside that checkout. All checkpoints,90 validation episode outcomes, raw training episodes/updates and earlier failures were retrieved and hash-verified from GitHub. Final test scenarios remain unused.

A first-failure counter reconciliation adds3 partial-macro primitives and2 selected robot decisions to the external aggregate; original bundle bytes remain unchanged. Stage resource totals and the active setup envelope are recorded in project_state.json. No quantum result is asserted.
