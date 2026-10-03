import torch

from cellsim.config import Config
from cellsim.model import ActorCritic
from cellsim.recorder import record_episode
from cellsim.storage import (
    append_metrics,
    list_replay_epochs,
    load_replay,
    read_metrics,
    replay_path,
    save_replay,
    truncate_metrics,
)


def model_for(cfg):
    torch.manual_seed(0)
    return ActorCritic(cfg.obs_dim)


def test_record_episode_structure():
    cfg = Config(episode_len=20)
    rep = record_episode(model_for(cfg), cfg, epoch=3)
    assert rep["epoch"] == 3
    assert rep["world"] == {"radius": 1.0, "cell_r": 0.03, "food_r": 0.015, "ray_len": 0.5, "n_rays": 16}
    assert len(rep["food_init"]) == 20
    assert len(rep["frames"]) == 21
    assert rep["frames"][0]["e"] == []
    for f in rep["frames"]:
        assert set(f) == {"p", "h", "r", "e"}
        assert len(f["r"]) == 16
        assert max(f["r"]) <= 0.5
    assert rep["score"] == sum(len(f["e"]) for f in rep["frames"])


def test_record_captures_eat_events():
    cfg = Config(episode_len=50, n_food=200)  # dense food: a forward-drifting cell must eat
    rep = record_episode(model_for(cfg), cfg, epoch=0)
    assert rep["score"] > 0
    idx, x, y = next(e for f in rep["frames"] for e in f["e"])
    assert 0 <= idx < 200 and (x * x + y * y) ** 0.5 <= 0.95 + 1e-3


def test_record_is_deterministic():
    cfg = Config(episode_len=20)
    m = model_for(cfg)
    assert record_episode(m, cfg, 1) == record_episode(m, cfg, 1)


def test_replay_roundtrip(tmp_path):
    cfg = Config(episode_len=5)
    rep = record_episode(model_for(cfg), cfg, epoch=5)
    save_replay(replay_path(tmp_path, 5), rep)
    assert load_replay(replay_path(tmp_path, 5)) == rep
    assert list_replay_epochs(tmp_path) == [5]
    assert not list((tmp_path / "replays").glob("*.tmp"))


def test_metrics_roundtrip_and_truncate(tmp_path):
    for e in range(5):
        append_metrics(tmp_path, {"epoch": e, "train_score": float(e)})
    with open(tmp_path / "metrics.jsonl", "a") as fh:
        fh.write('{"epoch": 5, "trai')  # simulated partial write
    assert [r["epoch"] for r in read_metrics(tmp_path)] == [0, 1, 2, 3, 4]
    truncate_metrics(tmp_path, 2)
    assert [r["epoch"] for r in read_metrics(tmp_path)] == [0, 1, 2]


def test_read_metrics_missing_file(tmp_path):
    assert read_metrics(tmp_path) == []
    assert list_replay_epochs(tmp_path) == []
