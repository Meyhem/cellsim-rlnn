import pytest
from fastapi.testclient import TestClient

from cellsim.config import Config
from cellsim.serve import create_app
from cellsim.storage import append_metrics, replay_path, save_replay

RUN = "20260101-000000"


@pytest.fixture
def client(tmp_path):
    run = tmp_path / RUN
    run.mkdir()
    (run / "config.json").write_text(Config().to_json())
    append_metrics(run, {"epoch": 0, "eval_score": 1})
    append_metrics(run, {"epoch": 1, "train_score": 2.5, "eval_score": 4})
    save_replay(replay_path(run, 0), {"epoch": 0, "score": 1, "frames": []})
    save_replay(replay_path(run, 1), {"epoch": 1, "score": 4, "frames": []})
    (tmp_path / "not-a-run").mkdir()
    return TestClient(create_app(tmp_path))


def test_list_runs(client):
    assert client.get("/api/runs").json() == [{"id": RUN, "latest_epoch": 1, "best_eval": 4}]


def test_metrics(client):
    rows = client.get(f"/api/runs/{RUN}/metrics").json()
    assert [r["epoch"] for r in rows] == [0, 1]


def test_replay_list_and_content(client):
    assert client.get(f"/api/runs/{RUN}/replays").json() == [0, 1]
    res = client.get(f"/api/runs/{RUN}/replays/1")
    assert res.status_code == 200
    assert res.headers["content-encoding"] == "gzip"
    assert res.json()["score"] == 4  # httpx transparently decodes gzip


@pytest.mark.parametrize("url", [
    "/api/runs/nope/metrics",
    "/api/runs/not-a-run/replays",
    f"/api/runs/{RUN}/replays/99",
    "/api/runs/../replays/0",
])
def test_not_found(client, url):
    assert client.get(url).status_code == 404


def test_viewer_index_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "CellSim" in res.text
    assert client.get("/app.js").status_code == 200
