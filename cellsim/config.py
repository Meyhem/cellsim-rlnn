from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # --- world ---
    dish_radius: float = 1.0
    cell_radius: float = 0.03
    food_radius: float = 0.015
    n_food: int = 20
    episode_len: int = 500
    spawn_radius: float = 0.95
    start_radius: float = 0.8
    respawn_min_dist: float = 0.1
    turn_rate: float = 0.2
    accel: float = 0.01
    drag: float = 0.85
    wall_penalty: float = 0.01
    n_rays: int = 16
    ray_len: float = 0.5
    # --- PPO ---
    n_envs: int = 1024
    rollout_len: int = 128
    epochs: int = 500
    gamma: float = 0.99
    gae_lambda: float = 0.95
    update_passes: int = 4
    minibatch_size: int = 8192
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 64
    # --- run ---
    record_every: int = 10
    eval_seed: int = 0
    seed: int = 1

    @property
    def obs_dim(self) -> int:
        return 2 * self.n_rays + 5

    @property
    def max_speed(self) -> float:
        return self.accel / (1.0 - self.drag)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> Config:
        data = json.loads(s)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
