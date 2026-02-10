# PPO 小游戏训练示例 (PyTorch)

使用 **PPO (Proximal Policy Optimization)** 在 Gymnasium 的 **CartPole-v1** 上训练，纯 PyTorch 实现。

## 环境说明

- **CartPole-v1**：平衡车，状态为 (位置, 速度, 杆角度, 角速度)，动作为左/右，目标保持杆不倒。
- 默认 100k 步可收敛到 500 分（满分）。

## 安装

```bash
pip install -r ppo/requirements.txt
```

## 训练

在项目根目录执行：

```bash
# 默认 CartPole-v1，10 万步
python ppo/train.py

# 自定义环境与步数
python ppo/train.py --env CartPole-v1 --steps 50000 --save-dir ppo/checkpoints
```

模型会周期保存到 `ppo/checkpoints/`。

## 评估

```bash
# 使用最新 checkpoint 评估（不弹窗）
python ppo/train.py --eval

# 指定模型并渲染
python ppo/train.py --eval --model ppo/checkpoints/ppo_CartPole-v1_500.pt --render
```

## 实现要点

- **ppo/ppo.py**
  - `ActorCritic`：共享 MLP + 策略头(离散 softmax) + 价值头
  - `compute_gae`：广义优势估计 (γ=0.99, λ=0.95)
  - `ppo_loss`：clip 目标 (ε=0.2) + value MSE + entropy 正则
- **ppo/train.py**
  - 每轮收集 2048 步 rollout，用 GAE 算 advantage/return
  - 每轮 10 个 epoch，mini-batch 64，PPO 更新
  - 支持 `--eval` 加载模型并跑若干局

## 换其他小游戏

Gymnasium 内置环境可直接换，例如：

```bash
python ppo/train.py --env LunarLander-v2 --steps 200000
```

若动作/观测空间与 CartPole 不同（如连续动作），需在 `ppo.py` 中改 `ActorCritic` 为连续动作版本（如高斯策略）。
