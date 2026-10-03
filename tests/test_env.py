import math

import torch

from cellsim.config import Config
from cellsim.env import BatchedDishEnv


def make_env(n=1, **kw):
    cfg = Config(**kw)
    env = BatchedDishEnv(n, cfg, device="cpu", seed=0)
    env.reset()
    return env, cfg


def place(env, pos, heading, food0=None, rest=(0.0, -0.9)):
    env.pos[:] = torch.tensor(pos)
    env.vel[:] = 0.0
    env.heading[:] = heading
    env.food[:] = torch.tensor(rest)
    if food0 is not None:
        env.food[:, 0] = torch.tensor(food0)


def test_observation_shape():
    env, cfg = make_env(n=4)
    assert env.observe().shape == (4, 37)


def test_ray_detects_food_ahead():
    env, cfg = make_env()
    place(env, (0.0, 0.0), 0.0, food0=(0.3, 0.0))
    obs = env.observe()
    expected = 1 - (0.3 - cfg.food_radius) / cfg.ray_len  # 0.43
    assert math.isclose(obs[0, 0].item(), expected, abs_tol=1e-5)
    assert torch.all(obs[0, 1:16] == 0)


def test_ray_index_follows_heading():
    env, cfg = make_env()
    place(env, (0.0, 0.0), -math.pi / 2, food0=(0.3, 0.0))  # food is 90° to the left
    obs = env.observe()
    assert obs[0, 4].item() > 0.4  # ray 4 = +90° relative to heading
    assert obs[0, 0].item() == 0


def test_wall_channel_distance():
    env, cfg = make_env()
    place(env, (0.7, 0.0), 0.0)
    obs = env.observe()
    assert math.isclose(obs[0, 16].item(), 1 - 0.3 / cfg.ray_len, abs_tol=1e-5)  # ray 0 wall
    assert obs[0, 16 + 8].item() == 0  # backward wall is 1.7 away


def test_eating_gives_reward_and_respawns():
    env, cfg = make_env()
    place(env, (0.0, 0.0), 0.0, food0=(0.04, 0.0))
    obs, reward, done, info = env.step(torch.tensor([[0.0, -1.0]]))  # zero thrust
    assert math.isclose(reward[0].item(), 1.0)
    assert info["eaten"][0, 0].item() is True
    assert env.score[0].item() == 1.0
    new = env.food[0, 0]
    assert new.norm().item() <= cfg.spawn_radius + 1e-6
    assert (new - env.pos[0]).norm().item() >= cfg.respawn_min_dist


def test_cell_stays_inside_dish_and_wall_penalized():
    env, cfg = make_env()
    place(env, (0.95, 0.0), 0.0)
    rewards = []
    for _ in range(50):
        _, r, _, info = env.step(torch.tensor([[0.0, 1.0]]))
        rewards.append(r[0].item())
        assert env.pos[0].norm().item() <= cfg.dish_radius - cfg.cell_radius + 1e-6
    assert math.isclose(min(rewards), -cfg.wall_penalty, abs_tol=1e-6)


def test_auto_reset_at_episode_end():
    env, cfg = make_env(n=2, episode_len=5)
    act = torch.zeros(2, 2)
    for _ in range(4):
        _, _, done, _ = env.step(act)
        assert not done.any()
    _, _, done, info = env.step(act)
    assert done.all()
    assert torch.all(env.t == 0)
    assert info["episode_scores"].numel() == 2


def test_randomized_time_marks_partial_episodes():
    cfg = Config()
    env = BatchedDishEnv(64, cfg, seed=0)
    env.reset(randomize_time=True)
    assert env.t.unique().numel() > 1
    assert torch.equal(env.partial, env.t > 0)


def test_same_seed_same_trajectory():
    cfg = Config()
    outs = []
    for _ in range(2):
        env = BatchedDishEnv(3, cfg, seed=42)
        obs = env.reset()
        g = torch.Generator().manual_seed(7)
        for _ in range(20):
            obs, _, _, _ = env.step(torch.rand(3, 2, generator=g) * 2 - 1)
        outs.append(obs)
    assert torch.equal(outs[0], outs[1])
