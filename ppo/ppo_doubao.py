import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

# 设备配置：优先用GPU，没有则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 1. 策略网络 + 价值网络 ======================
# 合并到一个网络里，简化代码（论文里是分开的，这里为了最简合并）
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        # 策略网络（输出动作的概率分布）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 离散动作空间用Softmax
        )
        # 价值网络（输出状态价值）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # 输出一个标量值
        )

    def get_action(self, state):
        """输入状态，输出动作和动作的对数概率（用于收集数据）"""
        state = torch.tensor(state, dtype=torch.float32).to(device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)  # 离散动作分布
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def evaluate(self, state, action):
        """评估给定状态和动作的对数概率、熵、状态价值（用于更新）"""
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()  # 熵正则化，鼓励探索
        value = self.critic(state)
        return log_prob, entropy, value

# ====================== 2. PPO核心算法 ======================
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, K_epochs=4):
        self.gamma = gamma  # 折扣因子
        self.clip_eps = clip_eps  # PPO Clip的ε（论文默认0.2）
        self.K_epochs = K_epochs  # 每批数据更新次数（论文默认4）
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 旧策略初始化为当前策略

    def compute_returns(self, rewards, dones, next_value):
        """计算折扣回报和优势函数（GAE简化版，适配最简代码）"""
        returns = []
        running_return = next_value
        # 逆序计算回报（从最后一步到第一步）
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return * (1 - dones[t])
            returns.insert(0, running_return)
        # 转换为张量
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        # 优势 = 回报 - 价值（简化版GAE）
        values = self.policy_old.critic(torch.tensor(states, dtype=torch.float32).to(device)).squeeze()
        advantages = returns - values.detach()
        # 优势归一化（提升稳定性）
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, states, actions, log_probs_old, rewards, dones):
        """PPO核心更新逻辑（对应论文Clip目标函数）"""
        # 转换为张量
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(device)
        
        # 计算下一步价值（用于回报计算）
        next_state = torch.tensor(states[-1], dtype=torch.float32).to(device)
        next_value = self.policy_old.critic(next_state).item()
        returns, advantages = self.compute_returns(rewards, dones, next_value)

        # 多次更新策略（论文K_epochs）
        for _ in range(self.K_epochs):
            # 计算新策略的对数概率、熵、价值
            log_probs, entropy, values = self.policy.evaluate(states, actions)
            # 计算策略比率 r(θ) = π_new / π_old
            ratio = torch.exp(log_probs - log_probs_old)
            
            # PPO Clip目标函数（论文核心公式）
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()  # 取min，限制更新幅度
            value_loss = nn.MSELoss()(values.squeeze(), returns)  # 价值网络损失
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()  # 熵正则化
            
            # 反向传播更新
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # 更新旧策略为新策略
        self.policy_old.load_state_dict(self.policy.state_dict())

# ====================== 3. 训练主逻辑 ======================
if __name__ == "__main__":
    # 环境配置（CartPole-v1是经典离散动作环境，适合入门）
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 初始化PPO
    ppo = PPO(state_dim, action_dim)

    # 训练参数
    max_episodes = 1000  # 最大训练回合数
    max_timesteps = 500  # 每回合最大步数
    print_interval = 10  # 每10回合打印一次结果

    # 训练循环
    for episode in range(max_episodes):
        state, _ = env.reset()
        log_probs_old = []
        states = []
        actions = []
        rewards = []
        dones = []
        total_reward = 0

        # 收集一回合的数据
        for t in range(max_timesteps):
            # 用旧策略选动作
            action, log_prob = ppo.policy_old.get_action(state)
            next_state, reward, done, trunc, _ = env.step(action)
            total_reward += reward

            # 存储数据
            states.append(state)
            actions.append(action)
            log_probs_old.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done or trunc:
                break

        # 用收集的数据更新PPO
        ppo.update(states, actions, log_probs_old, rewards, dones)

        # 打印训练进度
        if (episode + 1) % print_interval == 0:
            print(f"Episode: {episode+1}, Total Reward: {total_reward:.0f}")

        # 收敛判断（CartPole-v1满分500）
        if total_reward >= 500:
            print(f"训练完成！Episode {episode+1} 达到满分")
            break

    env.close()