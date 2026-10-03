from cellsim.config import Config


def test_defaults_match_spec():
    c = Config()
    assert c.obs_dim == 37
    assert abs(c.max_speed - 0.0667) < 1e-3
    assert (c.n_envs, c.rollout_len, c.epochs) == (1024, 128, 500)
    assert (c.n_food, c.episode_len, c.n_rays, c.ray_len) == (20, 500, 16, 0.5)
    assert (c.record_every, c.eval_seed) == (10, 0)


def test_json_roundtrip():
    c = Config(epochs=7, n_envs=3)
    assert Config.from_json(c.to_json()) == c


def test_from_json_ignores_unknown_keys():
    c = Config.from_json('{"epochs": 3, "bogus": 1}')
    assert c.epochs == 3
