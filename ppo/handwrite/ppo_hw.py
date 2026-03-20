from re import L
import torch
import sys
import torch.nn as nn
import gymnasium as gym
import torch.optim as optim
from pathlib import Path

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic(state)
        return action.item(), log_prob, value

    def evaluate(self, states, actions):
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.critic(states)
        return log_probs, entropy, values
        
class PPO:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, clip_eps=0.2, K_epochs=4):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.K_epochs = K_epochs
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.actor_critic_old = ActorCritic(state_dim, action_dim)
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())

    def compute_advantages(self, rewards, dones, values, next_value):
        advantages = []
        returns = []
        running_return = next_value
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns.insert(0, running_return)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(returns, dtype=torch.float32) - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, states, actions, log_probs, values, rewards, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs_old = torch.tensor(log_probs, dtype=torch.float32)
        values_old = torch.tensor(values, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        next_state = torch.tensor(states[-1], dtype=torch.float32)
        next_value = self.actor_critic_old.critic(next_state).item()
        advantages,returns = self.compute_advantages(rewards, dones, values_old, next_value)

        for _ in range(self.K_epochs):
            log_probs, entropy, values = self.actor_critic.evaluate(states, actions)
            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_lose = nn.MSELoss()(values.squeeze(), returns)
            total_loss = policy_loss + 0.5 * value_lose - 0.01 * entropy.mean()
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        
        self.actor_critic_old.load_state_dict(self.actor_critic.state_dict())

def play_human():
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    episode_num = 10
    for episode in range(episode_num):
        state, _ = env.reset()
        episode_over = False
        total_reward = 0
        while not episode_over:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = terminated or truncated
        print(f"Episode {episode} finished! Total reward: {total_reward}")

def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)
    episode_num = 100
    for episode in range(episode_num):
        state, _ = env.reset()
        episode_over = False
        total_reward = 0
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        while not episode_over:
            action, log_prob, value = ppo.actor_critic_old.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = terminated or truncated
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(terminated or truncated)
            state = next_state
        ppo.update(states, actions, log_probs, values, rewards, dones)
        print(f"Episode {episode} finished! Total reward: {total_reward}")

    # 使用脚本所在目录，避免从不同工作目录运行导致保存路径不存在
    save_dir = Path(__file__).resolve().parent
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "ppo_hw.pt"
    torch.save({"policy": ppo.actor_critic.state_dict(), "optimizer": ppo.optimizer.state_dict()}, save_path)

def play_ppo():
    env = gym.make("CartPole-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo = PPO(state_dim, action_dim)
    ckpt_path = Path(__file__).resolve().parent / "ppo_hw.pt"
    ppo.actor_critic.load_state_dict(torch.load(ckpt_path, map_location="cpu")["policy"])
    ppo.actor_critic.eval()
    state, _ = env.reset()
    episode_over = False
    total_reward = 0
    while not episode_over:
        with torch.no_grad():
            action, log_prob, value = ppo.actor_critic.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = terminated or truncated
            state = next_state
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    #train()
    play_ppo()