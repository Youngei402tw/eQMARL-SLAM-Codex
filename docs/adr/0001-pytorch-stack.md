# ADR 0001: PyTorch and PennyLane for the experimental implementation

**Status:** accepted for implementation planning; runtime compatibility remains a Stage 01 gate.  
**Decision date:** 2026-10-20.  
**Scope:** software architecture and reference migration, not a claim of improved learning performance.

## Decision

Use **PyTorch** for the shared actor, classical critics, optimization, and checkpoints; **PennyLane QNodes with the direct Torch interface** for quantum critics; and **Gymnasium** for the environment boundary. Target **Python 3.11** initially and begin with analytic CPU simulations of **4–8 total qubits**. Make **GTSAM an optional estimator backend**, so circuit checks and mapping experiments can run without importing it.

Keep the original TensorFlow/TensorFlow Quantum/Cirq implementation at its recorded source revision as an independent reference. Its old dependency environment belongs in a separate reference environment. It will not be imported by the new training package. This is a migration decision: the choice of tensor framework does not establish an eQMARL advantage.

The exact source revisions and scientific differences between the original implementation and the user's implementations are recorded in [REFERENCE_AUDIT.md](../REFERENCE_AUDIT.md).

## Why this stack

PennyLane documents that a QNode with `interface="torch"` accepts Torch tensors, returns Torch tensors, and participates in Torch automatic differentiation. That permits a small, inspectable `torch.nn.Module` containing a classical encoder, quantum circuit, and value head, with ordinary Torch optimizers. Use the direct interface first so parameter ownership, tensor shapes, dtype conversions, and differentiation remain explicit. [PennyLane Torch interface](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html)

PyTorch is a reasonable fit for the user's preference and for a common classical/quantum learner. The official MAPPO repository also provides a useful PyTorch algorithmic reference. Its legacy installation instructions are not a dependency specification for this project. We will implement the narrow learner needed here, with independent checks of returns, masks, and the PPO objective, rather than inherit the complete benchmark framework. [Official MAPPO implementation](https://github.com/marlbenchmark/on-policy)

Current PennyLane package metadata requires Python 3.11 or newer. The GTSAM package publishes a CPython 3.11 Linux x86-64 wheel, making Python 3.11 a sensible compatibility target. Those facts do not prove that every selected Torch, NumPy, PennyLane, and GTSAM version works together. Stage 01 must establish that with an actual clean installation and functional imports. [PennyLane package](https://pypi.org/project/pennylane/) · [GTSAM package](https://pypi.org/project/gtsam/)

CPU is the starting device because the planned circuits are small and accelerator overhead can dominate. PennyLane's Torch documentation specifically cautions about GPU overhead for small circuits. Profile the actual forward and backward workload before adopting a faster simulator or GPU; report wall time alongside environment steps. [PennyLane device discussion](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html)

## Package and backend boundaries

| Component | Contract | Dependency boundary |
| --- | --- | --- |
| Environment | Joint transition with declared agent axes, scalar team reward, separate termination and truncation, policy observations, and evaluation-only diagnostics | NumPy and Gymnasium |
| Estimator | Reset from a declared prior; update from odometry/sensors; return pose/map belief and uncertainty diagnostics | Lightweight toy backend first; GTSAM behind an optional adapter |
| Actor | Local observation/history to action distribution; one shared weight set with separate hidden state per agent | PyTorch |
| Critic | Training information allowed by the protocol to a declared team-value tensor | PyTorch; quantum variants additionally use PennyLane |
| Circuit reference | Explicit wire order, gates, parameters, state preparation, and observables to exact expectations | Independent NumPy oracle; optional isolated Cirq comparison |
| Evaluator | Frozen policies and registered test seeds to episode records and seed-level summaries | No training updates and no inference-time access to hidden truth |

The public quantum module should expose `forward(local_features) -> values` and a resource report. Its resolved configuration must include device, shots, differentiation method, real/complex dtype, total qubits, qubits per partition, layers, encoding, initial entanglement, gate topology, observables, head type, and noise model. Count classical encoder/readout parameters as well as circuit parameters.

Optional dependency groups should separate quantum simulation, SLAM estimation, analysis, development checks, and the isolated reference stack. Classical environment checks must not import TensorFlow, Cirq, or PennyLane. Avoid conversion through NumPy or detached tensors inside the differentiable encoder/circuit path; detach only where the learner intentionally forms fixed targets or logs diagnostics.

## Reference migration and subsequent improvements

Maintain distinct experiment identifiers for the following paths. Do not label a corrected path as a literal reproduction of the original.

| Path | What stays fixed | What the evidence can establish |
| --- | --- | --- |
| Original reference | Pinned TensorFlow/Cirq circuit, input encoding, readout, actor, objective, and selected original configuration | What that source revision computes; original training reproduction only if that protocol is actually rerun |
| Torch parity | Same selected circuit, parameters, inputs, readout, objective, and a fixed transition batch | Forward, gradient, and learner-term agreement across implementations |
| Corrected experiment | Every departure has a named configuration and a before/after check | Whether a particular correction or design change improves the registered task |
| Recurrent PPO experiment | One common recurrent actor/learner for all critic variants | Whether critic choice matters with the stronger, shared classical learner |

First port circuit and fixed-batch MAA2C/A2C semantics faithfully. Preserve the original single global observable and its weighted readout in this path. An affine value head, richer readout, revised timeout handling, recurrence, PPO, or a different feature encoder is a separate change. The original partite circuit contains an initial cross-partition entangling state followed by local partition operations; silently adding cross-partition gates in later layers changes the model being tested. [Original circuit](https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/eqmarl/circuits.py) · [Original learner](https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/eqmarl/algorithms/maa2c.py)

For the new partially observed tasks, a common recurrent actor and PPO/GAE are sensible subsequent choices. Each agent must have its own recurrent state even when weights are shared. All compared critics use the same actor architecture, rollout budget, action interface, observation permission, update schedule, and tuning budget. Any gain from recurrence or PPO must not be attributed to entanglement.

Gymnasium distinguishes task termination from external truncation. The learner must document whether the time limit is a true finite-horizon task boundary or an external cutoff; bootstrap masks and, where applicable, remaining-time observations must follow that decision. Do not collapse the two flags by default. For variable-duration macro-actions, use duration-consistent reward accumulation and discounting. [Gymnasium basic usage](https://gymnasium.farama.org/introduction/basic_usage/)

## Numerical and circuit parity gates

These are acceptance targets for small analytic fixtures, not results already obtained. If a backend cannot meet them, investigate the discrepancy and record any justified tolerance change before running comparisons.

| Check | Initial acceptance target |
| --- | --- |
| Bell preparation, wire order, control/target convention | State probabilities and Pauli expectations agree with the independent NumPy oracle to absolute tolerance `1e-8` in float64; independent state-norm error at most `1e-10` |
| Complete small circuit | Expectations agree with NumPy and the selected isolated reference to absolute tolerance `1e-8` in float64; separately record the looser tolerance required by a float32 TFQ reference |
| Derivatives | Torch gradients agree with central finite differences at nonsymmetric test points, initially `atol=1e-6`, `rtol=1e-3`; check finite-difference step-size convergence and record achieved error |
| Batching | Batched and per-sample evaluation and gradients agree within the same dtype-specific tolerance |
| Encoder gradient path | A declared nonsymmetric fixture gives finite gradients through the encoder and circuit; expected symmetry-induced zero gradients are handled separately |
| Objective parity | Same fixed transitions yield matching rewards, values, bootstraps, advantages, entropy, and actor/critic loss terms before optimizer differences are introduced |
| Checkpoint round trip | On the same device/software stack, restored parameters, optimizer, RNG, and recurrent state reproduce the next deterministic forward pass and update |

Use float64 real tensors and the corresponding complex precision for reference gates. Choose a training dtype only after measuring numerical error and runtime; make dtype conversion explicit. A float32 reference cannot be required to pass a float64 tolerance, and cross-platform bitwise identity is not the criterion for scientific replication.

### Bell-state sanity check that must precede learning

For a two-qubit Psi+ state, the exact initial correlations are `XX=+1`, `YY=+1`, and `ZZ=-1`, while each local Pauli expectation is zero. For `d` independent Bell pairs across two `d`-qubit partitions, each partition's reduced state is `I / 2^d`. Arbitrary local unitaries preserve it. Therefore every traceless observable confined to one partition, including that partition's Pauli parity, has zero expectation and zero ideal gradient as long as all subsequent operations remain local unitaries.

The acceptance fixture must demonstrate both this local-zero result and a nontrivial, input-sensitive joint correlator after suitable local rotations. Measuring only local expectations would discard the relevant correlations for this architecture. Adding later cross-partition gates, measurement-conditioned operations, or nonunitary channels changes the assumptions and requires separate tests. This follows directly from the reduced-state identity, not from a promised training advantage. The source state-preparation and partition structure are in the [original operations](https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/eqmarl/ops.py) and [circuits](https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/eqmarl/circuits.py).

## Analytic, finite-shot, and noisy experiments

| Mode | Initial implementation | Required interpretation |
| --- | --- | --- |
| Analytic development | PennyLane `default.qubit`, `shots=None`, Torch backpropagation | Exact simulator expectations with differentiable simulation |
| Alternative analytic simulator | For example, `lightning.qubit`, after compatible derivative and parity checks | A measured implementation optimization; unchanged scientific model |
| Finite-shot evaluation | Sample a frozen trained circuit at registered shot counts and measurement groups | Sensitivity of inference to measurement sampling |
| Finite-shot training | A backend-supported rule such as parameter shift, with execution/shot accounting | A distinct training experiment with noisy estimates and additional cost |
| Physical-noise simulation | Explicit channels, rates, placement, measurement model, and compatible differentiation on a small supported backend | Robustness under that stated noise model |

PennyLane documents that analytic backpropagation and finite-shot differentiation have different compatibility requirements. A circuit trained by exact backpropagation and evaluated with shot noise has not demonstrated finite-shot training. Shot noise alone is also not a device-noise model. Verify the exact chosen backend combination at runtime. [Differentiation interfaces](https://docs.pennylane.ai/en/stable/introduction/interfaces.html) · [Unsupported gradients](https://docs.pennylane.ai/en/stable/introduction/unsupported_gradients.html) · [Lightning device](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_qubit/device.html) · [Mixed-state device](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.default_mixed.DefaultMixed.html)

Record actual circuit executions, shots, compatible measurement groups, forward/backward time, peak memory, state preparation assumptions, and any communication needed to combine results. Reuploaded or shared parameters can make a naive two-executions-per-parameter estimate inaccurate. Simulating four to eight qubits does not establish a computational quantum advantage or a deployable quantum network.

## Stage 01 compatibility gate and locking policy

Before training, create a clean Python 3.11 CPU environment and execute all of the following:

1. Resolve and install the chosen Torch, NumPy, PennyLane, Gymnasium, test, and optional GTSAM packages. Record package indexes, platform, Python patch version, and any wheel/build constraints.
2. Import the actual modules. Run a tiny Torch encoder to QNode to scalar loss and backward; inspect shapes, finite values, and the intended encoder/circuit gradients.
3. Reset and step a tiny environment, validating observation/action spaces and the termination/truncation contract.
4. When the SLAM optional group is enabled, solve a tiny anchored GTSAM Pose2 problem with an actual measurement factor, obtain finite marginal covariance, and verify the estimator adapter's output units and frame convention.
5. Commit the real resolved lockfile and a machine-readable environment manifest only after these checks pass. The initial plan may state compatibility targets; it must not present invented pins or an untested lockfile as validated.

The original reference stack receives its own manifest, including the pinned original repository SHA and any local adaptations necessary to execute it. If the original environment cannot run, mark TFQ parity as blocked; independent NumPy parity can still proceed, but it is a narrower completed gate.

GTSAM's 4.3 release notes include Python/NumPy compatibility changes. Do not inherit a blanket `numpy<2` pin from older installation advice without checking the selected version. If an official wheel set cannot coexist with the learner, first isolate the estimator in a separate environment/process behind the same adapter. A validated EKF may be registered as a different benchmark backend, but must not silently replace GTSAM inside a named experiment. [GTSAM 4.3 release](https://github.com/borglab/gtsam/releases/tag/4.3.0)

## Consequences and limits

This design separates four possible causes of an apparent improvement: framework migration, environment/learner corrections, a stronger classical learning setup, and quantum critic structure. That separation is the reason to prefer it. It offers more auditable evidence and easier debugging; it does not guarantee higher reward.

The main engineering costs are maintaining a small independent circuit oracle, validating the Torch/quantum dtype boundary, keeping reference and corrected protocols separate, and recording simulator cost. GTSAM improves the ability to study explicit estimation factors and uncertainty, but its posterior remains conditional on the chosen noise and association models; estimator calibration still needs validation.

The original repository declares **CC BY 4.0**. Preserve source attribution, its license notice, and a modification record when porting or adapting its implementation. No upstream source code has been copied into this planning-only stage. [Original license](https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/LICENSE.md) · [Original package metadata](https://github.com/news-vt/eqmarl/blob/6b1b2e3817f661da81aad5e7188a362380dbdd1d/pyproject.toml)
