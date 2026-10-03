import torch

from cellsim.model import ActorCritic, RunningNorm


def test_running_norm_tracks_mean_and_var():
    torch.manual_seed(0)
    x = torch.randn(10000, 3) * 2 + 5
    n = RunningNorm(3)
    n.update(x[:5000])
    n.update(x[5000:])
    assert torch.allclose(n.mean, x.mean(0), atol=1e-3)
    assert torch.allclose(n.var, x.var(0, unbiased=False), rtol=1e-2)
    assert n(x).mean().abs().item() < 0.05


def test_running_norm_clips():
    n = RunningNorm(1)
    assert n(torch.tensor([[1e6]])).item() == 10.0


def test_actor_critic_shapes_and_consistency():
    torch.manual_seed(0)
    m = ActorCritic(37)
    obs = torch.randn(8, 37)
    a, logp, v = m.act(obs)
    assert a.shape == (8, 2) and logp.shape == (8,) and v.shape == (8,)
    logp2, ent, v2 = m.evaluate(obs, a)
    assert torch.allclose(logp, logp2, atol=1e-6)
    assert torch.allclose(v, v2, atol=1e-6)
    assert ent.shape == (8,)


def test_deterministic_action_small_at_init_and_clipped():
    torch.manual_seed(0)
    m = ActorCritic(37)
    det = m.act_deterministic(torch.randn(64, 37))
    assert det.abs().max().item() < 0.1
    with torch.no_grad():
        m.actor[-1].bias.fill_(5.0)
    assert m.act_deterministic(torch.randn(4, 37)).max().item() == 1.0


def test_state_dict_includes_normalizer():
    keys = ActorCritic(37).state_dict().keys()
    assert {"norm.mean", "norm.var", "norm.count", "log_std"} <= set(keys)
