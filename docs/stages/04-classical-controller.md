# Stage 04 — Common recurrent belief controller

Status: running. Dependency: `b0476fe8bb632389d62319003ef1e1c3c9312513`.
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
