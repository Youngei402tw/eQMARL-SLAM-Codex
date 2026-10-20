import numpy as np
import pytest
import torch

from eqmarl_slam.learner import corrected_a2c_terms, faithful_maa2c_terms


def fixture():
    p = torch.tensor([[[.15, .35, .5], [.8, .1, .1]], [[.4, .4, .2], [.2, .5, .3]], [[.7, .2, .1], [.1, .2, .7]]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[0, 2], [1, 0], [2, 1]])
    r = torch.tensor([[.4, -.2], [-1.2, .1], [2., .5]], dtype=torch.float64)
    v = torch.tensor([[.3], [-.2], [1.1]], dtype=torch.float64, requires_grad=True)
    nv = torch.tensor([[.9], [.5], [1.7]], dtype=torch.float64, requires_grad=True)
    done = torch.tensor([[False, False], [False, True], [False, False]])
    return p, a, r, v, nv, done


def test_faithful_terms_match_independent_numpy_arithmetic():
    p, a, r, v, nv, done = fixture()
    terms = faithful_maa2c_terms(p, a, r, v, nv, done, gamma=.83, alpha=.037)
    pn, an, rn, vn, nvn, dn = [x.detach().numpy() for x in [p, a, r, v, nv, done]]
    team_reward = np.array([[sum(row)] for row in rn])
    team_done = np.array([[any(row)] for row in dn], dtype=np.float64)
    bootstrap = (1 - team_done) * .83 * nvn
    target = team_reward + bootstrap
    advantage = target - vn
    chosen = np.array([[np.log(pn[t, j, an[t, j]]) for j in range(pn.shape[1])] for t in range(pn.shape[0])])
    entropy = np.mean([-sum(prob * np.log(prob) for prob in row[-1]) for row in pn])
    error = np.abs(target - vn)
    huber = np.where(error <= 1, .5 * error ** 2, error - .5).sum()
    expected = {"team_reward": team_reward, "team_done": team_done, "bootstrap": bootstrap, "target": target, "advantage": advantage, "chosen_log_probs": chosen, "entropy": entropy, "actor_loss": np.mean(-chosen * advantage) + .037 * entropy, "critic_loss": huber}
    for name, value in expected.items():
        np.testing.assert_allclose(terms[name].detach().numpy(), value, atol=1e-12, rtol=1e-12)
    # These are retained reference quirks; corrected code must be separate.
    terms["critic_loss"].backward()
    assert nv.grad is not None and torch.count_nonzero(nv.grad) > 0


def test_corrected_targets_and_advantages_detach_and_entropy_is_joint():
    p, a, r, v, nv, done = fixture()
    result = corrected_a2c_terms(p, a, r, v, nv, done, gamma=.83, alpha=.037)
    assert not result["target"].requires_grad
    assert not result["advantage"].requires_grad
    result["actor_loss"].backward(retain_graph=True)
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert v.grad is None and nv.grad is None
    result["critic_loss"].backward()
    assert v.grad is not None and nv.grad is None
    torch.testing.assert_close(result["entropy"], -(p * p.log()).sum(-1).mean())


def test_external_truncation_and_duration_discount_are_explicit():
    p, a, r, v, nv, done = fixture()
    terminal = torch.zeros_like(done)
    trunc = torch.zeros_like(done)
    trunc[0, 0] = True
    durations = torch.tensor([[3.], [2.], [1.]], dtype=torch.float64)
    external = corrected_a2c_terms(p, a, r, v, nv, terminal, truncated=trunc, durations=durations, gamma=.8)
    finite = corrected_a2c_terms(p, a, r, v, nv, terminal, truncated=trunc, durations=durations, gamma=.8, finite_horizon=True)
    torch.testing.assert_close(external["bootstrap"], .8 ** durations * nv.detach())
    assert finite["bootstrap"][0].item() == 0
    torch.testing.assert_close(finite["bootstrap"][1:], external["bootstrap"][1:])


def test_reject_ambiguous_shapes_and_invalid_durations():
    args = list(fixture())
    args[3] = args[3].squeeze(-1)
    with pytest.raises(ValueError, match="values"):
        faithful_maa2c_terms(*args)
    with pytest.raises(ValueError, match="whole primitive"):
        corrected_a2c_terms(*fixture(), durations=torch.full((3, 1), 1.5))
