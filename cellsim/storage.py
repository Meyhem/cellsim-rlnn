from __future__ import annotations

import gzip
import json
import os
import re
from pathlib import Path

REPLAY_RE = re.compile(r"^epoch_(\d+)\.json\.gz$")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def replay_path(run_dir: Path, epoch: int) -> Path:
    return Path(run_dir) / "replays" / f"epoch_{epoch:04d}.json.gz"


def checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"epoch_{epoch:04d}.pt"


def save_replay(path: Path, replay: dict) -> None:
    raw = json.dumps(replay, separators=(",", ":")).encode()
    atomic_write_bytes(path, gzip.compress(raw))


def load_replay(path: Path) -> dict:
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def list_replay_epochs(run_dir: Path) -> list[int]:
    d = Path(run_dir) / "replays"
    if not d.is_dir():
        return []
    return sorted(int(m.group(1)) for p in d.iterdir() if (m := REPLAY_RE.match(p.name)))


def append_metrics(run_dir: Path, row: dict) -> None:
    path = Path(run_dir) / "metrics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def read_metrics(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def truncate_metrics(run_dir: Path, max_epoch: int) -> None:
    rows = [r for r in read_metrics(run_dir) if r["epoch"] <= max_epoch]
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    atomic_write_bytes(Path(run_dir) / "metrics.jsonl", data)
