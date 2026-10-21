import numpy as np
import torch
from eqmarl_slam.critics import METHODS,make_critic


def test_entanglement_intervention_preserves_every_parameter():
    one,zero=make_critic('eqmarl',5),make_critic('quantum_noent',5)
    assert all(torch.equal(value,zero.state_dict()[name]) for name,value in one.state_dict().items())
    assert one.circuit.entanglement and not zero.circuit.entanglement


def test_common_compression_and_scalar_values_have_gradients():
    x=torch.tensor(np.random.default_rng(15).normal(size=(3,2,27)),dtype=torch.float64)
    for method in METHODS:
        model=make_critic(method,7)
        y=model(x)
        assert y.shape==(3,)
        y.sum().backward()
        assert model.encoder.weight.shape==(2,27)
        assert torch.isfinite(model.encoder.weight.grad).all()
        assert model.encoder.weight.grad.norm()>1e-10
