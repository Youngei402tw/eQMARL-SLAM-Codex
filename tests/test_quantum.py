"""Local implementation tests; scientific gate data are archived separately."""
import pennylane as qml
import pytest
import torch

from eqmarl_slam.quantum import (QuantumCircuit, SplitQuantumCritic,
                                ReferenceWeightedReadout, circuit_expectations)


def fixture(d=2):
    generator = torch.Generator().manual_seed(192)
    x = torch.randn(3, 2, d, generator=generator, dtype=torch.float64)
    theta = torch.randn(2, 2, d, 3, generator=generator, dtype=torch.float64)
    scales = torch.rand(2, 1, d, 1, generator=generator, dtype=torch.float64) + 0.5
    return x.requires_grad_(), theta.requires_grad_(), scales.requires_grad_()


@pytest.mark.parametrize("d", [1, 2, 4])
def test_batch_values_and_gradients_match_sample_loop(d):
    x, theta, scales = fixture(d)
    result = circuit_expectations(x, theta, scales)
    separate = torch.stack([circuit_expectations(row, theta, scales) for row in x])
    torch.testing.assert_close(result, separate, atol=1e-12, rtol=1e-12)
    grad_batch = torch.autograd.grad(result.sum(), (x, theta, scales), retain_graph=True)
    grad_loop = torch.autograd.grad(separate.sum(), (x, theta, scales))
    for actual, expected in zip(grad_batch, grad_loop):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        assert torch.isfinite(actual).all()
    assert grad_batch[0].abs().max() > 1e-5


@pytest.mark.parametrize("bell,expected", [("phi+", [1, -1, 1]), ("phi-", [-1, 1, 1]),
                                           ("psi+", [1, 1, -1]), ("psi-", [-1, -1, -1])])
def test_bell_correlations(bell, expected):
    circuit = QuantumCircuit(2, 1, 0, bell=bell, observables=("XX", "YY", "ZZ"))
    actual = circuit(torch.zeros(2, 1, dtype=torch.float64),
                     torch.zeros(2, 1, 1, 3, dtype=torch.float64),
                     torch.zeros(2, 0, 1, 1, dtype=torch.float64))
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64), atol=1e-12, rtol=0)


@pytest.mark.parametrize("d", [1, 2, 4])
def test_fully_paired_bell_local_marginals_zero(d):
    x, theta, scales = fixture(d)
    local = circuit_expectations(x, theta, scales, observables="local")
    parity = circuit_expectations(x, theta, scales, observables="partition")
    assert local.abs().max() < 1e-12
    assert parity.abs().max() < 1e-12
    joint = circuit_expectations(x, theta, scales)
    assert joint.std() > 1e-5


def test_encoder_circuit_head_gradients_and_device_execution_counts():
    torch.manual_seed(27)
    model = SplitQuantumCritic(5, parts=2, d=2)
    x = torch.randn(2, 2, 5, dtype=torch.float64)
    with qml.Tracker(model.circuit.device) as tracker:
        value = model(x)
        forward_executions = tracker.totals["executions"]
        value.square().sum().backward()
        assert tracker.totals["executions"] == forward_executions == 2
    for group in (model.encoder.parameters(), [model.theta], [model.scales], model.head.parameters()):
        assert sum(p.grad.abs().sum() for p in group) > 1e-8
    assert model.resource_report()["analytic_backward_additional_circuits"] == 0


def test_reference_rescaling_preserved():
    head = ReferenceWeightedReadout(beta=2)
    torch.testing.assert_close(head(torch.tensor([[-1.], [0.], [1.]], dtype=torch.float64)),
                               torch.tensor([[0.], [1.], [2.]], dtype=torch.float64))


def test_reject_implicit_dtype_or_shape_change():
    x, theta, scales = fixture()
    with pytest.raises(ValueError, match="float64"):
        circuit_expectations(x.float(), theta, scales)
    with pytest.raises(ValueError, match="scales"):
        circuit_expectations(x, theta, scales.squeeze(-1))
