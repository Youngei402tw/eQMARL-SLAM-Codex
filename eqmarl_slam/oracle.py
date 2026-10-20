"""Independent dense NumPy oracle for the pinned eQMARL split circuit.

Derived from news-vt/eqmarl 6b1b2e3 circuits.py and ops.py (CC BY 4.0).
This implementation constructs matrices; it shares no gate execution with the
PennyLane port. Wire 0 is the most significant computational basis bit.
The one-wire CZ ring is explicitly a no-op: upstream attempts an invalid self-CZ.
"""
from functools import reduce
import numpy as np

I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def single_matrix(gate, wire, n):
    return reduce(np.kron, [gate if j == wire else I for j in range(n)])


def controlled_matrix(control, target, n, kind):
    """Explicit big-endian matrix with column-as-input convention."""
    if control == target:
        raise ValueError("control and target must differ")
    matrix = np.zeros((2**n, 2**n), dtype=np.complex128)
    for column in range(2**n):
        active = (column >> (n - control - 1)) & 1
        if kind == "CNOT":
            row = column ^ (1 << (n - target - 1)) if active else column
            matrix[row, column] = 1
        elif kind == "CZ":
            target_bit = (column >> (n - target - 1)) & 1
            matrix[column, column] = -1 if active and target_bit else 1
        else:
            raise ValueError(kind)
    return matrix


def oracle_state(features, theta, scales, *, entanglement=True, bell="phi+", squash="atan"):
    features, theta, scales = [np.asarray(v, dtype=np.float64) for v in (features, theta, scales)]
    parts, d = features.shape
    layers = theta.shape[1] - 1
    if theta.shape != (parts, layers + 1, d, 3) or scales.shape != (parts, layers, d, 1):
        raise ValueError("inconsistent circuit shapes")
    if parts < 2 or layers < 1 or parts * d > 8:
        raise ValueError("oracle supports 2+ partitions, 1+ layers, at most 8 wires")
    if bell not in ("phi+", "phi-", "psi+", "psi-"):
        raise ValueError(bell)
    n = parts * d
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1

    def apply(gate, wire):
        nonlocal state
        # Tensor contraction remains independent of PennyLane's simulator.
        tensor = np.moveaxis(state.reshape((2,) * n), wire, 0)
        tensor = np.tensordot(gate, tensor, axes=(1, 0))
        state = np.moveaxis(tensor, 0, wire).reshape(-1)

    def controlled(control, target, kind):
        nonlocal state
        # Basis-permutation/phase implementation avoids dense O(4**n) storage.
        indices = np.arange(2**n)
        active = ((indices >> (n - control - 1)) & 1).astype(bool)
        if kind == "CNOT":
            rows = indices ^ (active.astype(int) << (n - target - 1))
            result = np.empty_like(state)
            result[rows] = state
            state = result
        else:
            active &= ((indices >> (n - target - 1)) & 1).astype(bool)
            state[active] *= -1

    if entanglement:
        for wire in range(d):
            if bell.endswith("-"):
                apply(X, wire)
            apply(H, wire)
            for part in range(parts - 1):
                c, t = part * d + wire, (part + 1) * d + wire
                if bell.startswith("psi"):
                    apply(X, t)
                controlled(c, t, "CNOT")

    for part in range(parts):
        for layer in range(layers + 1):
            for q in range(d):
                for axis, pauli in enumerate((X, Y, Z)):
                    angle = theta[part, layer, q, axis]
                    apply(np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * pauli, part * d + q)
            if layer == layers:
                break
            edges = list(zip(range(d - 1), range(1, d)))
            if d > 2:
                edges.append((0, d - 1))
            for c, t in edges:
                controlled(part * d + c, part * d + t, "CZ")
            for q in range(d):
                angle = scales[part, layer, q, 0] * features[part, q]
                if squash in ("atan", "arctan"):
                    angle = np.arctan(angle)
                elif squash == "tanh":
                    angle = np.tanh(angle)
                elif squash != "linear":
                    raise ValueError(squash)
                apply(np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * X, part * d + q)
    return state


def oracle_expectations(features, theta, scales, *, observables="global", **kwargs):
    state = oracle_state(features, theta, scales, **kwargs)
    n = int(np.log2(len(state)))
    indices = np.arange(len(state))
    signs = np.array([1 - 2 * ((indices >> (n - wire - 1)) & 1) for wire in range(n)])
    probabilities = abs(state)**2
    global_z = float(np.prod(signs, axis=0) @ probabilities)
    local = signs @ probabilities
    if observables == "global":
        return np.array([global_z])
    if observables == "local":
        return local
    if observables == "all":
        return np.r_[global_z, local]
    raise ValueError(observables)
