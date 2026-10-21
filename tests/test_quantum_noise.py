import numpy as np
import pennylane as qml
import pytest
import torch
from eqmarl_slam.quantum import QuantumCircuit


@pytest.fixture(autouse=True)
def mixed_backend_constant_precision():
    previous=torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try: yield
    finally: torch.set_default_dtype(previous)


def fixture():
    rng=np.random.default_rng(7001)
    return [torch.tensor(rng.normal(size=shape),dtype=torch.float64,requires_grad=True) for shape in [(2,2),(2,2,2,3),(2,1,2,1)]]


def test_mixed_channel_identity_and_seeded_sampling():
    x,theta,scales=fixture()
    clean=QuantumCircuit(2,2,1)(x,theta,scales)
    mixed=QuantumCircuit(2,2,1,noise=.05)(x,theta,scales)
    assert torch.allclose(mixed,clean*(1-4*.05/3)**4,atol=1e-10,rtol=0)
    a=QuantumCircuit(2,2,1,shots=64,seed=7002)
    b=QuantumCircuit(2,2,1,shots=64,seed=7002)
    assert torch.equal(a(x,theta,scales),b(x,theta,scales))


def test_noisy_finite_parameter_shift_tracks_backward_shots():
    x,theta,scales=fixture()
    circuit=QuantumCircuit(2,2,1,shots=64,noise=.05,seed=7003)
    with qml.Tracker(circuit.device) as monitor:
        circuit(x,theta,scales).sum().backward()
    assert monitor.totals['executions']>1
    assert monitor.totals['shots']==64*monitor.totals['executions']
    for tensor in [x,theta,scales]:
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.norm()>1e-10
