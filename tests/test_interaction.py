import numpy as np
import torch
from eqmarl_slam.interaction import InteractionCritic


def test_edge_is_symmetric_and_disabled_edge_preserves_local_terms():
    x=torch.tensor(np.random.default_rng(6001).normal(size=(3,2,27)),dtype=torch.float64)
    for kind in ['none','bilinear','quantum_noent','eqmarl']:
        torch.manual_seed(6002)
        model=InteractionCritic(kind)
        value=model(x)
        assert torch.allclose(value,model(x.flip(-2)),atol=1e-12,rtol=0)
        expected=model.local(model.encoder(x)).sum(-2).squeeze(-1)
        assert torch.allclose(model(x,torch.zeros(3)),expected,atol=1e-12,rtol=0)
        value.sum().backward()
        assert model.encoder.weight.grad.norm()>1e-10
