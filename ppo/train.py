"""
PPO 训练脚本：在 CartPole-v1 小游戏上训练
从项目根目录运行：python ppo/train.py  或  python -m ppo.train
"""
import argparse
import sys
from pathlib import Path

# 保证从项目根可导入 ppo
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo import ActorCritic, compute_gae, ppo_loss


def collect_rollout(env, policy, device, n_steps):
    """收集一条 rollout：n_steps 步"""
    obs_buf = []
    action_buf = []
    reward_buf = []
    done_buf = []
    log_prob_buf = []
    value_buf = []

    obs, _ = env.reset()
    episode_reward = 0.0

    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value, _ = policy.get_action(obs_t)

        obs_buf.append(obs)
        action_buf.append(action[0])
        log_prob_buf.append(log_prob.item())
        value_buf.append(value.item())

        next_obs, reward, terminated, truncated, _ = env.step(int(action[0]))
        done = terminated or truncated
        reward_buf.append(reward)
        done_buf.append(done)
        episode_reward += reward

        if done:
            obs, _ = env.reset()
            episode_reward = 0.0
        else:
            obs = next_obs

    next_value = 0.0
    next_done = 1.0
    if not done:
        with torch.no_grad():
            _, next_value, _, _ = policy.get_action(
                torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            )
        next_value = next_value.item()
        next_done = 0.0

    advantages, returns = compute_gae(
        np.array(reward_buf, dtype=np.float32),
        np.array(value_buf, dtype=np.float32),
        np.array(done_buf, dtype=np.float32),
        next_value,
        next_done,
        gamma=0.99,
        lam=0.95,
    )

    return {
        "obs": torch.as_tensor(np.array(obs_buf), dtype=torch.float32, device=device),
        "actions": torch.as_tensor(np.array(action_buf), dtype=torch.long, device=device),
        "old_log_probs": torch.as_tensor(np.array(log_prob_buf), dtype=torch.float32, device=device),
        "advantages": torch.as_tensor(advantages, dtype=torch.float32, device=device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=device),
    }


def train(
    env_id: str = "CartPole-v1",
    total_timesteps: int = 100_000,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    lr: float = 3e-4,
    save_dir: str = "ppo/checkpoints",
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")
    policy = ActorCritic(obs_dim, act_dim, hidden_dim=64).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    n_updates = total_timesteps // n_steps

    for update in range(n_updates):
        rollout = collect_rollout(env, policy, device, n_steps)

        # 多轮 mini-batch 更新
        indices = np.arange(n_steps)
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_steps, batch_size):
                end = start + batch_size
                mb_idx = indices[start:end]

                loss, info = ppo_loss(
                    policy,
                    rollout["obs"][mb_idx],
                    rollout["actions"][mb_idx],
                    rollout["old_log_probs"][mb_idx],
                    rollout["advantages"][mb_idx],
                    rollout["returns"][mb_idx],
                    clip_eps=0.2,
                    value_coef=0.5,
                    entropy_coef=0.01,
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

        if (update + 1) % 10 == 0:
            print(
                f"Update {update + 1}/{n_updates}  "
                f"policy_loss={info['policy_loss']:.4f}  "
                f"value_loss={info['value_loss']:.4f}  "
                f"entropy={info['entropy']:.4f}"
            )
            # 每 10 轮保存一次（10 万步约 48 轮，按 50 轮保存会一次都不触发）
            path = Path(save_dir) / f"ppo_{env_id}_{update + 1}.pt"
            torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict()}, path)
            print(f"  Saved {path}")

    # 训练结束再保存一份最终模型
    final_path = Path(save_dir) / f"ppo_{env_id}_final.pt"
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict()}, final_path)
    print(f"Saved final model: {final_path}")
    env.close()
    return policy


def evaluate(env_id: str, policy_path: str, n_episodes: int = 5, render: bool = False):
    """加载模型并评估"""
    env = gym.make(env_id, render_mode="human" if render else None)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = ActorCritic(obs_dim, act_dim, hidden_dim=64).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device)["policy"])
    policy.eval()

    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        while True:
            with torch.no_grad():
                action, _, _, _ = policy.get_action(
                    torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
                    deterministic=True,
                )
            obs, reward, terminated, truncated, _ = env.step(int(action[0]))
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
    env.close()
    print(f"Eval mean return: {np.mean(returns):.1f} ± {np.std(returns):.1f}")
    return returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", help="Gymnasium 环境 ID")
    parser.add_argument("--steps", type=int, default=100_000, help="总步数")
    parser.add_argument("--save-dir", default="ppo/checkpoints", help="模型保存目录")
    parser.add_argument("--eval", action="store_true", help="评估模式")
    parser.add_argument("--model", default="", help="评估时模型路径")
    parser.add_argument("--render", action="store_true", help="评估时是否渲染")
    args = parser.parse_args()

    if args.eval:
        if not args.model:
            args.model = str(Path(args.save_dir).glob("*.pt").__next__())
        evaluate(args.env, args.model, n_episodes=5, render=args.render)
    else:
        train(
            env_id=args.env,
            total_timesteps=args.steps,
            save_dir=args.save_dir,
        )
