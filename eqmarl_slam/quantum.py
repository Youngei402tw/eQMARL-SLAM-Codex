"""Torch-interface port of the pinned eQMARL partite circuit.

Architecture source: news-vt/eqmarl at
6b1b2e3817f661da81aad5e7188a362380dbdd1d, circuits.py, ops.py and
layers.py (CC BY 4.0). This port uses PennyLane, float64, explicit tensor
axes and independent readout modules. It does not port the optimizer.

Wires are partition-major and wire 0 is the most significant basis bit.
Variational rotations act in Rx, Ry, Rz order. The only inter-partition
gates are the initial Bell/GHZ preparation. A one-wire partition has no
CZ ring: this is an explicit diagnostic extension because the upstream
helper attempts a two-wire gate on the same wire in that case.
"""

from __future__ import annotations

from functools import lru_cache

import pennylane as qml
import torch
from torch import nn


def _observable_words(parts: int, d: int, observables):
    wires = parts * d
    if observables == "global":
        return ("Z" * wires,)
    if observables == "local":
        return tuple("I" * i + "Z" + "I" * (wires - i - 1) for i in range(wires))
    if observables == "partition":
        return tuple("I" * (p * d) + "Z" * d + "I" * ((parts - p - 1) * d) for p in range(parts))
    if observables == "all":
        return _observable_words(parts, d, "global") + _observable_words(parts, d, "local")
    if isinstance(observables, (tuple, list)) and observables:
        words = tuple(observables)
        if any(len(word) != wires or not set(word).issubset("IXYZ") for word in words):
            raise ValueError("Observable words must contain I/X/Y/Z for every wire")
        return words
    raise ValueError(f"Unsupported observables: {observables!r}")


class QuantumCircuit:
    """Reusable circuit and public device for an external ``qml.Tracker``.

    Batches deliberately use a sample loop, making execution counts
    unambiguous. Analytic backpropagation reuses the forward state and
    does not execute parameter-shift circuits in the backward pass.
    """

    def __init__(self, parts: int, d: int, layers: int, *, entanglement=True,
                 bell="phi+", observables="global", squash="atan",
                 shots=None, diff_method=None):
        if parts < 1 or d < 1 or layers < 0 or parts * d > 8:
            raise ValueError("Require positive parts/d, nonnegative layers and at most 8 wires")
        if bell not in ("phi+", "phi-", "psi+", "psi-"):
            raise ValueError(f"Unsupported Bell preparation: {bell}")
        if squash not in ("atan", "arctan", "linear", "tanh"):
            raise ValueError(f"Unsupported squash transformation: {squash}")
        if shots is not None and (isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0):
            raise ValueError("shots must be None or a positive integer")
        self.parts, self.d, self.layers = parts, d, layers
        self.entanglement, self.bell, self.squash = entanglement, bell, squash
        self.words = _observable_words(parts, d, observables)
        self.shots = shots
        self.diff_method = diff_method or ("backprop" if shots is None else "parameter-shift")
        if shots is not None and self.diff_method == "backprop":
            raise ValueError("Finite shots require a compatible differentiation method")
        self.device = qml.device("default.qubit", wires=parts * d, shots=shots)

        @qml.qnode(self.device, interface="torch", diff_method=self.diff_method)
        def qnode(features, theta, scales):
            if self.entanglement:
                for q in range(d):
                    if bell.endswith("-"):
                        qml.PauliX(wires=q)
                    qml.Hadamard(wires=q)
                    for p in range(parts - 1):
                        target = (p + 1) * d + q
                        if bell.startswith("psi"):
                            qml.PauliX(wires=target)
                        qml.CNOT(wires=[p * d + q, target])
            for p in range(parts):
                for layer in range(layers + 1):
                    for q in range(d):
                        wire = p * d + q
                        qml.RX(theta[p, layer, q, 0], wires=wire)
                        qml.RY(theta[p, layer, q, 1], wires=wire)
                        qml.RZ(theta[p, layer, q, 2], wires=wire)
                    if layer == layers:
                        continue
                    for q in range(d - 1):
                        qml.CZ(wires=[p * d + q, p * d + q + 1])
                    if d > 2:
                        qml.CZ(wires=[p * d, (p + 1) * d - 1])
                    for q in range(d):
                        angle = scales[p, layer, q, 0] * features[p, q]
                        if squash in ("atan", "arctan"):
                            angle = torch.atan(angle)
                        elif squash == "tanh":
                            angle = torch.tanh(angle)
                        qml.RX(angle, wires=p * d + q)
            measurements = []
            paulis = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
            for word in self.words:
                factors = [paulis[letter](wires=i) for i, letter in enumerate(word) if letter != "I"]
                observable = qml.prod(*factors) if factors else qml.Identity(0)
                measurements.append(qml.expval(observable))
            return tuple(measurements)

        self.qnode = qnode

    def __call__(self, features, theta, scales):
        features = torch.as_tensor(features)
        if features.shape[-2:] != (self.parts, self.d):
            raise ValueError(f"features must end in {(self.parts, self.d)}")
        if tuple(theta.shape) != (self.parts, self.layers + 1, self.d, 3):
            raise ValueError("theta must have axes (parts,layers+1,d,3)")
        if tuple(scales.shape) != (self.parts, self.layers, self.d, 1):
            raise ValueError("scales must have axes (parts,layers,d,1)")
        if features.dtype != torch.float64 or theta.dtype != torch.float64 or scales.dtype != torch.float64:
            raise ValueError("Reference circuit requires explicit torch.float64 tensors")
        if features.device.type != "cpu" or theta.device.type != "cpu" or scales.device.type != "cpu":
            raise ValueError("This registered circuit uses CPU tensors")
        batch_shape = features.shape[:-2]
        flat = features.reshape(-1, self.parts, self.d)
        if flat.shape[0] == 0:
            raise ValueError("Empty circuit batches are not supported")
        values = [torch.stack(self.qnode(sample, theta, scales)) for sample in flat]
        return torch.stack(values).reshape(*batch_shape, len(self.words))

    def resource_report(self):
        parts, d, layers = self.parts, self.d, self.layers
        depths = [0] * (parts * d)

        def schedule(*wires):
            next_depth = max(depths[wire] for wire in wires) + 1
            for wire in wires:
                depths[wire] = next_depth

        if self.entanglement:
            for q in range(d):
                if self.bell.endswith("-"):
                    schedule(q)
                schedule(q)
                for p in range(parts - 1):
                    target = (p + 1) * d + q
                    if self.bell.startswith("psi"):
                        schedule(target)
                    schedule(p * d + q, target)
        for p in range(parts):
            for layer in range(layers + 1):
                for q in range(d):
                    for _ in range(3):
                        schedule(p * d + q)
                if layer == layers:
                    continue
                for q in range(d - 1):
                    schedule(p * d + q, p * d + q + 1)
                if d > 2:
                    schedule(p * d, (p + 1) * d - 1)
                for q in range(d):
                    schedule(p * d + q)
        prep = (d + d * (parts - 1)) if self.entanglement else 0
        if self.entanglement and self.bell.endswith("-"):
            prep += d
        if self.entanglement and self.bell.startswith("psi"):
            prep += d * (parts - 1)
        ring_edges = 0 if d == 1 else (1 if d == 2 else d)
        return {
            "device": "default.qubit", "total_qubits": parts * d,
            "qubits_per_partition": d, "parts": parts, "layers": layers,
            "real_dtype": "float64", "complex_dtype": "complex128",
            "wire_order": "partition_major_wire0_most_significant",
            "rotation_order": ["RX", "RY", "RZ"],
            "encoding": f"{self.squash}(scale*feature) then RX",
            "initial_entanglement": self.bell if self.entanglement else "none",
            "later_inter_partition_gates": 0,
            "one_wire_ring": "no_op_diagnostic_extension",
            "gate_counts": {"state_preparation": prep,
                            "variational_rotations": parts * (layers + 1) * d * 3,
                            "local_CZ": parts * layers * ring_edges,
                            "encoding_RX": parts * layers * d},
            "gate_count_convention": "logical RX/RY/RZ/H/X/CNOT/CZ; no hardware transpilation",
            "logical_depth": max(depths),
            "logical_depth_convention": "earliest scheduling preserving per-wire gate order; measurement excluded",
            "observables": list(self.words), "shots_per_execution": self.shots,
            "diff_method": self.diff_method, "noise": None,
            "execution_accounting": "qml.Tracker device executions; sample-loop batches",
            "analytic_backward_additional_circuits": 0 if self.diff_method == "backprop" else None,
            "physical_shots_sampled": self.shots is not None,
        }


@lru_cache(maxsize=64)
def make_circuit(parts, d, layers, *, entanglement=True, bell="phi+",
                 observables="global", squash="atan", shots=None, diff_method=None):
    """Cached analytic circuit; callers may track ``.device`` explicitly."""
    return QuantumCircuit(parts, d, layers, entanglement=entanglement, bell=bell,
                          observables=observables, squash=squash, shots=shots,
                          diff_method=diff_method)


def circuit_expectations(features, theta, scales, *, entanglement=True,
                         bell="phi+", observables="global", squash="atan", tracker=None):
    """Evaluate (...,parts,d) inputs, returning (...,observables).

    ``tracker`` may be a dictionary, which receives actual forward device
    execution counters. To count finite-shot backward executions too, use
    ``QuantumCircuit`` and a ``qml.Tracker`` spanning forward and backward.
    """
    parts, layer_count, d, rotations = theta.shape
    if rotations != 3:
        raise ValueError("Variational rotations must have a final axis of length 3")
    if isinstance(observables, list):
        observables = tuple(observables)
    circuit = make_circuit(parts, d, layer_count - 1, entanglement=entanglement,
                           bell=bell, observables=observables, squash=squash)
    if tracker is None:
        return circuit(features, theta, scales)
    with qml.Tracker(circuit.device) as observed:
        values = circuit(features, theta, scales)
    for key in ("executions", "simulations", "shots"):
        tracker[key] = tracker.get(key, 0) + observed.totals.get(key, 0)
    return values


class ReferenceWeightedReadout(nn.Module):
    """Pinned critic's ``beta * w * (1+expectation)/2`` readout.

    The rescaling is preserved, including its nonnegative output at the
    original positive initialization. The weight itself is unconstrained.
    """
    def __init__(self, n_observables=1, beta=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_observables, dtype=torch.float64))
        self.beta = float(beta)

    def forward(self, expectations):
        return self.beta * self.weight * (1.0 + expectations) / 2.0


class CorrectedAffineReadout(nn.Linear):
    """Separately named unconstrained affine team-value head."""
    def __init__(self, n_observables=1):
        super().__init__(n_observables, 1, bias=True, dtype=torch.float64)


class SplitQuantumCritic(nn.Module):
    """Shared local compression followed by a split circuit and value head.

    This learned compression is a corrected experiment component, not a
    claim to reproduce the reference's task-specific observation adapter.
    Inputs have axes (...,parts,input_dim); outputs have axes (...,1).
    """
    def __init__(self, input_dim, parts=2, d=2, layers=1, *, entanglement=True,
                 bell="phi+", head="affine", trainable_scales=True, squash="atan"):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d, dtype=torch.float64)
        self.theta = nn.Parameter(torch.rand(parts, layers + 1, d, 3, dtype=torch.float64) * torch.pi)
        self.scales = nn.Parameter(torch.ones(parts, layers, d, 1, dtype=torch.float64), requires_grad=trainable_scales)
        if head not in ("affine", "reference"):
            raise ValueError("head must be affine or reference")
        self.head = CorrectedAffineReadout() if head == "affine" else ReferenceWeightedReadout()
        self.head_type = head
        self.circuit = QuantumCircuit(parts, d, layers, entanglement=entanglement,
                                      bell=bell, squash=squash)

    def forward(self, local_features):
        if local_features.shape[-2] != self.circuit.parts:
            raise ValueError("The partition axis must match the configured robot count")
        compressed = self.encoder(local_features.to(dtype=self.theta.dtype))
        return self.head(self.circuit(compressed, self.theta, self.scales))

    def resource_report(self):
        report = self.circuit.resource_report()
        report.update({"head": self.head_type,
                       "parameter_counts": {"encoder": sum(p.numel() for p in self.encoder.parameters()),
                                            "circuit": self.theta.numel() + (self.scales.numel() if self.scales.requires_grad else 0),
                                            "readout": sum(p.numel() for p in self.head.parameters()),
                                            "total_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad)},
                       "encoder_scope": "shared pointwise local compression; critic receives permitted joint training inputs"})
        return report
