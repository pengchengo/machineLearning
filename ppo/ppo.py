"""
PPO (Proximal Policy Optimization) 算法实现 - PyTorch
包含：Actor-Critic 网络、GAE 优势估计、Clipped 目标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """共享特征 + 策略头(离散动作) + 价值头"""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        feat = self.shared(obs)
        logits = self.actor(feat)
        value = self.critic(feat).squeeze(-1)
        return logits, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.cpu().numpy(), log_prob, value, dist.entropy()

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, value.squeeze(-1), entropy


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    next_done: float,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """广义优势估计 GAE"""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_val = next_value
            next_non_terminal = 1.0 - next_done
        else:
            next_val = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae * next_non_terminal
    returns = advantages + values
    return advantages, returns


def ppo_loss(
    policy: ActorCritic,
    obs: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    计算 PPO 损失：policy clip + value MSE + entropy bonus
    """
    log_prob, value, entropy = policy.evaluate(obs, actions)

    # Clipped surrogate
    ratio = (log_prob - old_log_probs).exp()
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surr1 = ratio * adv
    surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(value, returns)
    entropy_loss = -entropy.mean()

    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    info = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.mean().item(),
    }
    return loss, info
