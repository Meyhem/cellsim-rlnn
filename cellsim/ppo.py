from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from cellsim.config import Config
from cellsim.env import BatchedDishEnv
from cellsim.model import ActorCritic, RunningNorm


class RewardScaler(nn.Module):
    """Scales rewards by the running std of the discounted return."""

    def __init__(self, n_envs: int, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("ret", torch.zeros(n_envs))
        self.stats = RunningNorm(1)

    @torch.no_grad()
    def forward(self, reward: Tensor, done: Tensor) -> Tensor:
        self.ret.mul_(self.gamma).add_(reward)
        self.stats.update(self.ret[:, None])
        scaled = reward / torch.sqrt(self.stats.var[0] + 1e-8)
        self.ret.mul_(1.0 - done.float())
        return scaled


@dataclass
class Rollout:
    obs: Tensor
    actions: Tensor
    logprobs: Tensor
    rewards: Tensor
    dones: Tensor
    values: Tensor
    last_value: Tensor


@torch.no_grad()
def collect_rollout(
    env: BatchedDishEnv, model: ActorCritic, scaler: RewardScaler, obs: Tensor, n_steps: int
) -> tuple[Rollout, Tensor, list[float]]:
    n, d, dev = env.n_envs, obs.shape[-1], obs.device
    buf_obs = torch.zeros(n_steps, n, d, device=dev)
    buf_act = torch.zeros(n_steps, n, 2, device=dev)
    buf_logp = torch.zeros(n_steps, n, device=dev)
    buf_rew = torch.zeros(n_steps, n, device=dev)
    buf_done = torch.zeros(n_steps, n, device=dev)
    buf_val = torch.zeros(n_steps, n, device=dev)
    scores: list[Tensor] = []
    for t in range(n_steps):
        action, logp, value = model.act(obs)
        next_obs, reward, done, info = env.step(action)
        buf_obs[t] = obs
        buf_act[t] = action
        buf_logp[t] = logp
        buf_val[t] = value
        buf_rew[t] = scaler(reward, done)
        buf_done[t] = done.float()
        scores.append(info["episode_scores"])
        obs = next_obs
    last_value = model.value(obs)
    roll = Rollout(buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val, last_value)
    return roll, obs, torch.cat(scores).tolist()


@torch.no_grad()
def compute_gae(
    rewards: Tensor, values: Tensor, dones: Tensor, last_value: Tensor, gamma: float, lam: float
) -> tuple[Tensor, Tensor]:
    n_steps = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(last_value)
    for t in reversed(range(n_steps)):
        next_value = last_value if t == n_steps - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    obs: Tensor,
    actions: Tensor,
    old_logprobs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    cfg: Config,
) -> dict[str, float]:
    batch = obs.shape[0]
    mb = min(cfg.minibatch_size, batch)
    eps = cfg.clip_eps
    stats: dict[str, list[float]] = defaultdict(list)
    for _ in range(cfg.update_passes):
        perm = torch.randperm(batch, device=obs.device)
        for start in range(0, batch, mb):
            idx = perm[start : start + mb]
            logp, entropy, value = model.evaluate(obs[idx], actions[idx])
            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            log_ratio = logp - old_logprobs[idx]
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
            value_loss = 0.5 * ((value - returns[idx]) ** 2).mean()
            ent = entropy.mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(ent.item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > eps).float().mean().item())
    return {k: sum(v) / len(v) for k, v in stats.items()}
