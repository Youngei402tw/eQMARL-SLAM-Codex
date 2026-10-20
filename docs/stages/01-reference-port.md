# Stage 01 — Reference port and model contracts

Status: running. Dependency: Stage 00 completion `1c14d2ad833be18b1d46d0d402ab915fe4c19e26`.
Branch: `stage/01-reference-port`, created from that exact commit on 2026-10-20.
The user's execution instruction authorizes the complete recorded plan; the
Stage 00 planning-only scope is superseded for this session.

## Registered question and falsifier

Does the float64 Torch/PennyLane port compute the pinned split circuit and
maintain differentiable inputs and parameters? The smallest falsifier is a
nondegenerate fixed two-partition fixture whose parity expectation disagrees
with an independent NumPy statevector or whose gradient disagrees with central
finite differences. A failed gate blocks subsequent scientific comparisons.

## Deliverables and gates

- Clean Python 3.11 CPU runtime and real dependency locks; separate classical-only installation.
- Pinned reference source with attribution; independent NumPy oracle, PennyLane
  port, reference weighted readout, and separately named affine readout.
- 2/4/8-qubit expectations, state norms, finite-difference gradients, batching,
  local Bell marginals, fixed-batch learner terms, information/shape contracts,
  checkpoint/RNG/optimizer round trip, and throughput/resource measurements.
- Bundle verifier, CLI, CPU CI, complete diagnostic bundles and remote receipts.

Float64 expectation tolerance: absolute `1e-8`; NumPy norm error: `1e-10`.
Gradient tolerance: `atol=1e-6, rtol=1e-3`, with step convergence from `1e-4`
to `1e-5`. The original float32 TFQ comparison, if runnable, uses separately
registered `atol=2e-5, rtol=2e-5` (single-precision simulator/rotation errors);
it cannot replace the float64 gate.

Diagnostic cap: 3,600 seconds per family; stage cumulative cap: 86,400 active
execution seconds; one CPU thread per diagnostic process. There is no training
budget use in this stage. Exact simulation samples no physical shots; reported
zero sampled shots is not a hardware-cost claim.

## Reference boundaries

Source: `news-vt/eqmarl@6b1b2e3817f661da81aad5e7188a362380dbdd1d`.
Wires are partition-major, wire 0 the most significant basis bit. Each local
block applies RX, RY, RZ, ring CZ, then RX of `atan(scale*feature)`, followed by
the final variational block. Inter-partition gates occur only during initial
Bell/GHZ preparation. The faithful readout is `beta*w*(1+parity)/2`.

The upstream one-wire ring creates an invalid same-wire CZ. The two-qubit
diagnostic explicitly defines that local ring as empty; 4/8-qubit fixtures use
the upstream topology unchanged. This is a diagnostic extension, not exact
execution of that invalid upstream case.

The faithful one-step MAA2C arithmetic retains last-agent-only entropy,
positive entropy contribution, and undetached bootstrap/advantage paths.
Corrected targets, entropy treatment, timeout/duration semantics, and recurrent
models are separately named. No historical training outcome is reproduced by
these numerical fixtures.

## Execution and publication

Run definitions and outcomes are indexed in `experiments/01/index.json`.
Source/config commits are published and read back before reportable execution.
Every outcome, including failures and installation diagnostics, is retained.
Completion and remaining budgets will be recorded after actual execution.
