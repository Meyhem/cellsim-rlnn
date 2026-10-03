from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles

from cellsim.storage import list_replay_epochs, read_metrics, replay_path

VIEWER_DIR = Path(__file__).resolve().parent.parent / "viewer"


def create_app(runs_dir: Path) -> FastAPI:
    runs_dir = Path(runs_dir)
    app = FastAPI(title="CellSim viewer")

    def run_ids() -> list[str]:
        if not runs_dir.is_dir():
            return []
        ids = [p.name for p in runs_dir.iterdir() if (p / "config.json").is_file()]
        return sorted(ids, reverse=True)

    def run_dir(run_id: str) -> Path:
        if run_id not in run_ids():
            raise HTTPException(404, f"unknown run {run_id!r}")
        return runs_dir / run_id

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        out = []
        for rid in run_ids():
            rows = read_metrics(runs_dir / rid)
            evals = [r["eval_score"] for r in rows if r.get("eval_score") is not None]
            out.append({
                "id": rid,
                "latest_epoch": max((r["epoch"] for r in rows), default=None),
                "best_eval": max(evals, default=None),
            })
        return out

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str) -> list[dict]:
        return read_metrics(run_dir(run_id))

    @app.get("/api/runs/{run_id}/replays")
    def replays(run_id: str) -> list[int]:
        return list_replay_epochs(run_dir(run_id))

    @app.get("/api/runs/{run_id}/replays/{epoch}")
    def replay(run_id: str, epoch: int) -> Response:
        path = replay_path(run_dir(run_id), epoch)
        if not path.is_file():
            raise HTTPException(404, f"no replay for epoch {epoch}")
        return Response(
            content=path.read_bytes(),
            media_type="application/json",
            headers={"Content-Encoding": "gzip", "Cache-Control": "no-cache"},
        )

    if VIEWER_DIR.is_dir():
        app.mount("/", StaticFiles(directory=VIEWER_DIR, html=True), name="viewer")
    return app


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Serve the CellSim replay viewer")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    print(f"CellSim viewer: http://{args.host}:{args.port}")
    uvicorn.run(create_app(Path(args.runs_dir)), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
