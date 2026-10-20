import random

import numpy as np
import pytest
import torch

from eqmarl_slam.checkpoint import load_checkpoint, save_checkpoint


class RecurrentFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = torch.nn.GRUCell(3, 4, dtype=torch.float64)
        self.head = torch.nn.Linear(4, 1, dtype=torch.float64)

    def forward(self, x, h):
        h = self.cell(x, h)
        return self.head(h), h


def step(model, optimizer, h):
    x = torch.randn(2, 3, dtype=torch.float64) + random.random() + float(np.random.randn())
    y = torch.randn(2, 1, dtype=torch.float64)
    optimizer.zero_grad()
    value, hidden = model(x, h)
    loss = (value - y).square().mean()
    loss.backward()
    optimizer.step()
    return value.detach().clone(), hidden.detach().clone(), loss.detach().clone()


def test_complete_rng_optimizer_recurrent_checkpoint_next_update(tmp_path):
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    model = RecurrentFixture()
    optimizer = torch.optim.Adam(model.parameters(), lr=.012)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3)
    h = torch.zeros(2, 4, dtype=torch.float64)
    _, h, _ = step(model, optimizer, h)
    scheduler.step()
    path = tmp_path / "state.pt"
    digest = save_checkpoint(path, model, optimizer, recurrent_state=h, progress={"updates": 1}, normalization={"mean": torch.tensor(.17), "count": 7}, scheduler=scheduler, environment_state={"steps": 4})
    expected = step(model, optimizer, h)
    expected_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()

    # Constructing the fresh model consumes Torch RNG, so restoration must
    # occur after construction and before the next stochastic update.
    restored = RecurrentFixture()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.5)
    restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=3)
    state = load_checkpoint(path, restored, restored_optimizer, expected_sha256=digest, scheduler=restored_scheduler)
    actual = step(restored, restored_optimizer, state["recurrent_state"])
    for x, y in zip(expected, actual):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected_weights[name], atol=0, rtol=0)
    for key, value in expected_optimizer["state"].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, restored_optimizer.state_dict()["state"][key][field], atol=0, rtol=0)
    assert state["progress"] == {"updates": 1}
    assert state["environment_state"] == {"steps": 4}
    assert state["normalization"]["count"] == 7
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_tampered_checkpoint_rejected_before_loading(tmp_path):
    model = RecurrentFixture()
    optimizer = torch.optim.Adam(model.parameters())
    path = tmp_path / "state.pt"
    digest = save_checkpoint(path, model, optimizer, recurrent_state=None, progress={})
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        load_checkpoint(path, model, optimizer, expected_sha256=digest)


def test_checkpoint_rejects_opaque_state(tmp_path):
    model = RecurrentFixture()
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(TypeError, match="plain"):
        save_checkpoint(tmp_path / "state.pt", model, optimizer, recurrent_state=object(), progress={})
