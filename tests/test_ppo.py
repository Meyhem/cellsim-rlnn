import torch

from cellsim.config import Config
from cellsim.env import BatchedDishEnv
from cellsim.model import ActorCritic
from cellsim.ppo import RewardScaler, collect_rollout, compute_gae, ppo_update


def test_gae_matches_hand_computation():
    rewards = torch.tensor([[1.0], [0.0], [2.0]])
    values = torch.tensor([[0.5], [0.4], [0.3]])
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    last_value = torch.tensor([1.0])
    adv, ret = compute_gae(rewards, values, dones, last_value, gamma=0.9, lam=0.8)
    assert torch.allclose(adv[:, 0], torch.tensor([0.572, -0.4, 2.6]), atol=1e-6)
    assert torch.allclose(ret[:, 0], torch.tensor([1.072, 0.0, 2.9]), atol=1e-6)


def test_reward_scaler_resets_return_on_done():
    s = RewardScaler(n_envs=2, gamma=0.99)
    out = s(torch.tensor([1.0, 1.0]), torch.tensor([False, True]))
    assert torch.isfinite(out).all()
    assert s.ret[0].item() == 1.0 and s.ret[1].item() == 0.0


def test_collect_rollout_shapes():
    cfg = Config(episode_len=10)
    env = BatchedDishEnv(4, cfg, seed=0)
    model = ActorCritic(cfg.obs_dim)
    scaler = RewardScaler(4, cfg.gamma)
    obs = env.reset()
    roll, next_obs, scores = collect_rollout(env, model, scaler, obs, n_steps=12)
    assert roll.obs.shape == (12, 4, 37)
    assert roll.actions.shape == (12, 4, 2)
    for name in ("logprobs", "rewards", "dones", "values"):
        assert getattr(roll, name).shape == (12, 4)
    assert roll.last_value.shape == (4,)
    assert next_obs.shape == (4, 37)
    assert len(scores) == 4  # every env finished exactly one 10-step episode


def test_ppo_update_moves_policy_toward_positive_advantage():
    torch.manual_seed(0)
    cfg = Config(update_passes=4, minibatch_size=256)
    model = ActorCritic(37)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    obs = torch.randn(256, 37)
    actions, old_logp, _ = model.act(obs)
    adv = torch.cat([torch.ones(128), -torch.ones(128)])
    returns = torch.zeros(256)
    stats = ppo_update(model, opt, obs, actions, old_logp, adv, returns, cfg)
    assert set(stats) == {"policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"}
    with torch.no_grad():
        new_logp, _, _ = model.evaluate(obs, actions)
    delta = new_logp - old_logp
    assert delta[:128].mean().item() > 0
    assert delta[128:].mean().item() < 0
