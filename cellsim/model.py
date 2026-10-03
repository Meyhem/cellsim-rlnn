from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class RunningNorm(nn.Module):
    """Running mean/variance normalizer (parallel-variance update)."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.clip = clip
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        b_mean = x.mean(0)
        b_var = x.var(0, unbiased=False)
        b_count = x.shape[0]
        tot = self.count + b_count
        delta = b_mean - self.mean
        new_mean = self.mean + delta * (b_count / tot)
        m2 = self.var * self.count + b_var * b_count + delta**2 * (self.count * b_count / tot)
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / tot)
        self.count.fill_(tot.item())

    def forward(self, x: Tensor) -> Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


def _layer(n_in: int, n_out: int, gain: float) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 64):
        super().__init__()
        g = math.sqrt(2)
        self.norm = RunningNorm(obs_dim)
        self.actor = nn.Sequential(
            _layer(obs_dim, hidden, g), nn.Tanh(),
            _layer(hidden, hidden, g), nn.Tanh(),
            _layer(hidden, act_dim, 0.01),
        )
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden, g), nn.Tanh(),
            _layer(hidden, hidden, g), nn.Tanh(),
            _layer(hidden, 1, 1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs: Tensor) -> Normal:
        mu = self.actor(self.norm(obs))
        return Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, obs: Tensor) -> Tensor:
        return self.critic(self.norm(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def evaluate(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d = self.dist(obs)
        return d.log_prob(actions).sum(-1), d.entropy().sum(-1), self.value(obs)

    @torch.no_grad()
    def act_deterministic(self, obs: Tensor) -> Tensor:
        return self.actor(self.norm(obs)).clamp(-1.0, 1.0)
