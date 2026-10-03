import torch

from cellsim.config import Config
from cellsim.storage import list_replay_epochs, read_metrics
from cellsim.train import main, train

QUIET = lambda *_: None  # noqa: E731


def tiny_cfg(**kw):
    base = dict(n_envs=8, rollout_len=16, epochs=2, record_every=1, episode_len=20, minibatch_size=64)
    base.update(kw)
    return Config(**base)


def test_train_writes_artifacts(tmp_path):
    run = tmp_path / "run"
    train(tiny_cfg(), run, "cpu", log=QUIET)
    assert list_replay_epochs(run) == [0, 1, 2]
    rows = read_metrics(run)
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert "eval_score" in rows[0]
    assert {"policy_loss", "entropy", "sps", "eval_score"} <= set(rows[1])
    ck = torch.load(run / "checkpoints" / "latest.pt")
    assert ck["epoch"] == 2
    assert (run / "checkpoints" / "epoch_0002.pt").exists()
    assert Config.from_json((run / "config.json").read_text()) == tiny_cfg()


def test_record_every_controls_recordings(tmp_path):
    run = tmp_path / "run"
    train(tiny_cfg(epochs=5, record_every=2), run, "cpu", log=QUIET)
    assert list_replay_epochs(run) == [0, 2, 4, 5]  # 5 = final epoch


def test_resume_continues_epochs(tmp_path):
    run = tmp_path / "run"
    train(tiny_cfg(), run, "cpu", log=QUIET)
    train(tiny_cfg(epochs=3), run, "cpu", resume=True, log=QUIET)
    assert [r["epoch"] for r in read_metrics(run)] == [0, 1, 2, 3]
    assert list_replay_epochs(run) == [0, 1, 2, 3]


def test_main_cli(tmp_path):
    run = main(["--epochs", "1", "--envs", "8", "--record-every", "1",
                "--device", "cpu", "--runs-dir", str(tmp_path)])
    assert run.parent == tmp_path
    assert list_replay_epochs(run) == [0, 1]
