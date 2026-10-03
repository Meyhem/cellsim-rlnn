from __future__ import annotations

import argparse
import io
import signal
import time
from dataclasses import asdict
from pathlib import Path

import torch

from cellsim.config import Config
from cellsim.env import BatchedDishEnv
from cellsim.model import ActorCritic
from cellsim.ppo import RewardScaler, collect_rollout, compute_gae, ppo_update
from cellsim.recorder import record_episode
from cellsim.storage import (
    append_metrics,
    atomic_write_bytes,
    checkpoint_path,
    replay_path,
    save_replay,
    truncate_metrics,
)


def make_run_dir(root: Path) -> Path:
    root = Path(root)
    name = time.strftime("%Y%m%d-%H%M%S")
    path, k = root / name, 1
    while path.exists():
        k += 1
        path = root / f"{name}-{k}"
    path.mkdir(parents=True)
    return path


def _save_checkpoint(run_dir: Path, epoch: int, model, optimizer, scaler, cfg: Config) -> None:
    buf = io.BytesIO()
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": asdict(cfg),
        },
        buf,
    )
    data = buf.getvalue()
    atomic_write_bytes(checkpoint_path(run_dir, epoch), data)
    atomic_write_bytes(Path(run_dir) / "checkpoints" / "latest.pt", data)


def _record(run_dir: Path, model, cfg: Config, epoch: int) -> int:
    replay = record_episode(model, cfg, epoch)
    save_replay(replay_path(run_dir, epoch), replay)
    return replay["score"]


def train(cfg: Config, run_dir: Path, device: str, resume: bool = False, log=print) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(run_dir / "config.json", cfg.to_json().encode())
    torch.manual_seed(cfg.seed)

    env = BatchedDishEnv(cfg.n_envs, cfg, device=device, seed=cfg.seed)
    model = ActorCritic(cfg.obs_dim, 2, cfg.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    scaler = RewardScaler(cfg.n_envs, cfg.gamma).to(device)

    if resume:
        ck = torch.load(run_dir / "checkpoints" / "latest.pt", map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"] + 1
        truncate_metrics(run_dir, ck["epoch"])
        log(f"resumed {run_dir} at epoch {start}")
    else:
        score = _record(run_dir, model, cfg, 0)
        append_metrics(run_dir, {"epoch": 0, "eval_score": score})
        _save_checkpoint(run_dir, 0, model, optimizer, scaler, cfg)
        log(f"epoch    0 | eval {score:3d} (untrained) | run {run_dir}")
        start = 1

    stop = {"flag": False}

    def on_sigint(signum, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        log("Ctrl+C: finishing this epoch and saving (press again to abort)")

    prev_handler = signal.signal(signal.SIGINT, on_sigint)
    try:
        obs = env.reset(randomize_time=True)
        for epoch in range(start, cfg.epochs + 1):
            t0 = time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = cfg.lr * (1.0 - (epoch - 1) / cfg.epochs)

            roll, obs, scores = collect_rollout(env, model, scaler, obs, cfg.rollout_len)
            adv, ret = compute_gae(roll.rewards, roll.values, roll.dones, roll.last_value,
                                   cfg.gamma, cfg.gae_lambda)
            d = roll.obs.shape[-1]
            stats = ppo_update(model, optimizer, roll.obs.reshape(-1, d), roll.actions.reshape(-1, 2),
                               roll.logprobs.reshape(-1), adv.reshape(-1), ret.reshape(-1), cfg)
            model.norm.update(roll.obs.reshape(-1, d))

            elapsed = time.perf_counter() - t0
            row = {
                "epoch": epoch,
                "train_score": (sum(scores) / len(scores)) if scores else None,
                "episodes": len(scores),
                **stats,
                "sps": cfg.rollout_len * cfg.n_envs / elapsed,
                "time": elapsed,
            }
            if epoch % cfg.record_every == 0 or epoch == cfg.epochs or stop["flag"]:
                row["eval_score"] = _record(run_dir, model, cfg, epoch)
                _save_checkpoint(run_dir, epoch, model, optimizer, scaler, cfg)
            append_metrics(run_dir, row)

            ts = f"{row['train_score']:6.2f}" if row["train_score"] is not None else "     -"
            ev = f" | eval {row['eval_score']:3d}" if "eval_score" in row else ""
            log(f"epoch {epoch:4d} | train {ts}{ev} | ent {stats['entropy']:.2f} "
                f"| kl {stats['approx_kl']:.4f} | sps {row['sps']:,.0f}")
            if stop["flag"]:
                break
    finally:
        signal.signal(signal.SIGINT, prev_handler)


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Train a CellSim forager with PPO")
    p.add_argument("--epochs", type=int)
    p.add_argument("--envs", type=int)
    p.add_argument("--record-every", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--resume", default=None, help="run directory to continue")
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.resume:
        run_dir = Path(args.resume)
        cfg = Config.from_json((run_dir / "config.json").read_text())
        if args.envs is not None and args.envs != cfg.n_envs:
            p.error("--envs cannot change when resuming")
    else:
        cfg = Config()
        run_dir = make_run_dir(Path(args.runs_dir))
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.envs is not None:
        cfg.n_envs = args.envs
    if args.record_every is not None:
        cfg.record_every = args.record_every
    if args.seed is not None:
        cfg.seed = args.seed

    train(cfg, run_dir, device, resume=bool(args.resume))
    return run_dir


if __name__ == "__main__":
    main()
