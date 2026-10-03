from __future__ import annotations

import math

import torch
from torch import Tensor

from cellsim.config import Config

TWO_PI = 2.0 * math.pi


class BatchedDishEnv:
    """N independent petri dishes simulated as batched tensors."""

    def __init__(self, n_envs: int, cfg: Config, device="cpu", seed: int = 0, auto_reset: bool = True):
        self.n_envs = n_envs
        self.cfg = cfg
        self.device = torch.device(device)
        self.auto_reset = auto_reset
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        n, f, dev = n_envs, cfg.n_food, self.device
        self.pos = torch.zeros(n, 2, device=dev)
        self.vel = torch.zeros(n, 2, device=dev)
        self.heading = torch.zeros(n, device=dev)
        self.food = torch.zeros(n, f, 2, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.last_action = torch.zeros(n, 2, device=dev)
        self.score = torch.zeros(n, device=dev)
        self.partial = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ray_offsets = torch.arange(cfg.n_rays, device=dev, dtype=torch.float32) * (TWO_PI / cfg.n_rays)

    # ---------- randomness ----------
    def _rand(self, *shape) -> Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _sample_disk(self, shape: tuple, radius: float) -> Tensor:
        r = radius * torch.sqrt(self._rand(*shape))
        a = TWO_PI * self._rand(*shape)
        return torch.stack((r * torch.cos(a), r * torch.sin(a)), dim=-1)

    def _respawn_food(self, mask: Tensor) -> None:
        """Move food where mask (N,F) is True to new spots >= respawn_min_dist from the cell."""
        cfg = self.cfg
        new = self.food.clone()
        todo = mask.clone()
        cand = new
        for _ in range(20):
            if not todo.any():
                break
            cand = self._sample_disk(tuple(mask.shape), cfg.spawn_radius)
            ok = (cand - self.pos[:, None]).norm(dim=-1) >= cfg.respawn_min_dist
            take = todo & ok
            new = torch.where(take[..., None], cand, new)
            todo = todo & ~take
        new = torch.where(todo[..., None], cand, new)  # practically unreachable fallback
        self.food = new

    # ---------- reset ----------
    def _reset_envs(self, idx: Tensor) -> None:
        n = idx.numel()
        if n == 0:
            return
        self.pos[idx] = self._sample_disk((n,), self.cfg.start_radius)
        self.vel[idx] = 0.0
        self.heading[idx] = self._rand(n) * TWO_PI
        self.t[idx] = 0
        self.last_action[idx] = 0.0
        self.score[idx] = 0.0
        self.partial[idx] = False
        mask = torch.zeros(self.n_envs, self.cfg.n_food, dtype=torch.bool, device=self.device)
        mask[idx] = True
        self._respawn_food(mask)

    def reset(self, randomize_time: bool = False) -> Tensor:
        self._reset_envs(torch.arange(self.n_envs, device=self.device))
        if randomize_time:
            self.t = torch.randint(0, self.cfg.episode_len, (self.n_envs,), generator=self.gen, device=self.device)
            self.partial = self.t > 0
        return self.observe()

    # ---------- sensing ----------
    def cast_rays(self) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        ang = self.heading[:, None] + self.ray_offsets[None]  # (N,K)
        d = torch.stack((torch.cos(ang), torch.sin(ang)), dim=-1)  # (N,K,2)

        rel = self.food - self.pos[:, None]  # (N,F,2)
        b = torch.einsum("nkd,nfd->nkf", d, rel)  # projection along ray
        dist2 = (rel**2).sum(-1)[:, None, :]  # (N,1,F)
        perp2 = dist2 - b**2
        r2 = cfg.food_radius**2
        t_hit = b - torch.sqrt(torch.clamp(r2 - perp2, min=0.0))
        valid = (perp2 <= r2) & (t_hit >= 0) & (t_hit <= cfg.ray_len)
        t_food = torch.where(valid, t_hit, torch.full_like(t_hit, math.inf)).min(dim=-1).values

        od = (d * self.pos[:, None]).sum(-1)  # (N,K)
        c = (self.pos**2).sum(-1, keepdim=True) - cfg.dish_radius**2  # (N,1), <= 0 inside
        t_wall = -od + torch.sqrt(torch.clamp(od**2 - c, min=0.0))
        return t_food, t_wall

    def observe(self) -> Tensor:
        cfg = self.cfg
        t_food, t_wall = self.cast_rays()
        food_ch = (1.0 - t_food / cfg.ray_len).clamp(min=0.0)
        wall_ch = (1.0 - t_wall / cfg.ray_len).clamp(min=0.0)
        c, s = torch.cos(self.heading), torch.sin(self.heading)
        fwd = (self.vel[:, 0] * c + self.vel[:, 1] * s) / cfg.max_speed
        lat = (-self.vel[:, 0] * s + self.vel[:, 1] * c) / cfg.max_speed
        time_left = 1.0 - self.t.float() / cfg.episode_len
        proprio = torch.stack((fwd, lat, self.last_action[:, 0], self.last_action[:, 1], time_left), dim=-1)
        return torch.cat((food_ch, wall_ch, proprio), dim=-1)

    # ---------- dynamics ----------
    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, dict]:
        cfg = self.cfg
        a = actions.to(self.device).clamp(-1.0, 1.0)
        self.heading = torch.remainder(self.heading + a[:, 0] * cfg.turn_rate, TWO_PI)
        thrust = (a[:, 1] + 1.0) * 0.5
        direction = torch.stack((torch.cos(self.heading), torch.sin(self.heading)), dim=-1)
        self.vel = cfg.drag * self.vel + cfg.accel * thrust[:, None] * direction
        self.pos = self.pos + self.vel

        limit = cfg.dish_radius - cfg.cell_radius
        dist = self.pos.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        normal = self.pos / dist
        out = dist.squeeze(-1) > limit
        self.pos = torch.where(out[:, None], normal * limit, self.pos)
        radial = (self.vel * normal).sum(-1, keepdim=True)
        self.vel = torch.where(out[:, None] & (radial > 0), self.vel - radial * normal, self.vel)
        wall = out.float()

        eaten = (self.food - self.pos[:, None]).norm(dim=-1) < cfg.cell_radius + cfg.food_radius
        n_eaten = eaten.sum(-1).float()
        if eaten.any():
            self._respawn_food(eaten)
        self.score += n_eaten
        reward = n_eaten - cfg.wall_penalty * wall
        self.last_action = a
        self.t += 1

        done = self.t >= cfg.episode_len
        info = {
            "eaten": eaten,
            "wall": wall,
            "episode_scores": self.score[done & ~self.partial].clone(),
        }
        if self.auto_reset and done.any():
            self._reset_envs(done.nonzero().squeeze(-1))
        return self.observe(), reward, done, info
