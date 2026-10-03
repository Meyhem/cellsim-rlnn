from __future__ import annotations

import torch

from cellsim.config import Config
from cellsim.env import BatchedDishEnv
from cellsim.model import ActorCritic


def _r(x: float) -> float:
    return round(float(x), 4)


def _frame(env: BatchedDishEnv, events: list) -> dict:
    t_food, t_wall = env.cast_rays()
    rays = torch.minimum(t_food[0], t_wall[0]).clamp(max=env.cfg.ray_len)
    return {
        "p": [_r(env.pos[0, 0]), _r(env.pos[0, 1])],
        "h": _r(env.heading[0]),
        "r": [_r(d) for d in rays.tolist()],
        "e": events,
    }


@torch.no_grad()
def record_episode(model: ActorCritic, cfg: Config, epoch: int) -> dict:
    """Play one deterministic episode on the fixed eval dish and return a replay dict."""
    device = next(model.parameters()).device
    env = BatchedDishEnv(1, cfg, device=device, seed=cfg.eval_seed, auto_reset=False)
    obs = env.reset()
    food_init = [[_r(x), _r(y)] for x, y in env.food[0].tolist()]
    frames = [_frame(env, [])]
    for _ in range(cfg.episode_len):
        obs, _, _, info = env.step(model.act_deterministic(obs))
        idx = info["eaten"][0].nonzero().squeeze(-1).tolist()
        events = [[i, _r(env.food[0, i, 0]), _r(env.food[0, i, 1])] for i in idx]
        frames.append(_frame(env, events))
    return {
        "epoch": epoch,
        "score": int(env.score[0].item()),
        "world": {
            "radius": cfg.dish_radius,
            "cell_r": cfg.cell_radius,
            "food_r": cfg.food_radius,
            "ray_len": cfg.ray_len,
            "n_rays": cfg.n_rays,
        },
        "food_init": food_init,
        "frames": frames,
    }
