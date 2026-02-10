# PyTorch CNN 手写数字识别

使用PyTorch实现的MNIST手写数字识别项目，包含简单的CNN模型和更高级的CNN模型。

## 项目结构

```
pytorchcnn/
├── model.py          # 模型定义（SimpleCNN, AdvancedCNN）
├── train.py          # 训练脚本
├── test.py           # 测试脚本
├── predict.py        # 单张图片预测脚本
├── requirements.txt  # 依赖包
└── README.md         # 说明文档
```

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

#### 使用简单模型训练（推荐初学者）
```bash
python train.py --model simple --epochs 10 --batch_size 64
```

#### 使用高级模型训练（性能更好）
```bash
python train.py --model advanced --epochs 10 --batch_size 64
```

#### 完整参数示例
```bash
python train.py \
    --model advanced \
    --batch_size 128 \
    --epochs 20 \
    --lr 0.001 \
    --data_dir ./data \
    --save_dir ./checkpoints \
    --log_dir ./runs
```

#### 恢复训练
```bash
python train.py --model simple --resume ./checkpoints/simple_epoch_5.pth
```

### 2. 测试模型

```bash
python test.py --model simple --checkpoint ./checkpoints/simple_best.pth
```

### 3. 预测单张图片

```bash
python predict.py --model simple --checkpoint ./checkpoints/simple_best.pth --image your_image.png
```

## 模型说明

### SimpleCNN（简单模型）
- **结构**: Conv2d(8) -> MaxPool -> Conv2d(16) -> MaxPool -> FC(128) -> FC(10)
- **参数量**: 约 50,000
- **特点**: 轻量级，训练快速，适合学习和快速验证
- **预期准确率**: 98%+

### AdvancedCNN（高级模型）
- **结构**: Conv2d(32) -> BN -> MaxPool -> Conv2d(64) -> BN -> MaxPool -> Conv2d(128) -> BN -> MaxPool -> FC(256) -> FC(10)
- **参数量**: 约 1,000,000
- **特点**: 使用BatchNorm和Dropout，性能更好，更稳定
- **预期准确率**: 99%+

## 训练参数说明

- `--model`: 模型类型，'simple' 或 'advanced'
- `--batch_size`: 批次大小，默认64
- `--epochs`: 训练轮数，默认10
- `--lr`: 学习率，默认0.001
- `--data_dir`: 数据保存目录，默认'./data'
- `--save_dir`: 模型保存目录，默认'./checkpoints'
- `--log_dir`: TensorBoard日志目录，默认'./runs'
- `--resume`: 恢复训练的检查点路径

## 查看训练过程

使用TensorBoard查看训练曲线：

```bash
tensorboard --logdir ./runs
```

然后在浏览器打开 `http://localhost:6006`

## 数据集

项目使用PyTorch内置的MNIST数据集，首次运行会自动下载到 `data_dir` 目录。

- **训练集**: 60,000张图片
- **测试集**: 10,000张图片
- **图片大小**: 28x28 灰度图
- **类别数**: 10 (数字0-9)

## 预期结果

### SimpleCNN
- 训练10个epoch后，测试准确率通常能达到 **98%+**
- 训练时间（CPU）: 约5-10分钟
- 训练时间（GPU）: 约1-2分钟

### AdvancedCNN
- 训练10个epoch后，测试准确率通常能达到 **99%+**
- 训练时间（CPU）: 约15-20分钟
- 训练时间（GPU）: 约2-3分钟

## 文件说明

- **checkpoints/**: 保存训练好的模型
  - `{model}_best.pth`: 验证集上表现最好的模型
  - `{model}_epoch_{n}.pth`: 每5个epoch保存的检查点

- **runs/**: TensorBoard日志文件

- **data/**: MNIST数据集（自动下载）

## 常见问题

### 1. 内存不足
- 减小 `batch_size`，例如改为32或16

### 2. 训练速度慢
- 如果有GPU，确保PyTorch正确识别GPU
- 增加 `num_workers`（在代码中修改）

### 3. 准确率不理想
- 增加训练轮数 `--epochs`
- 尝试使用 `advanced` 模型
- 检查数据预处理是否正确

### 4. 如何预测自己的手写数字图片？
- 确保图片是灰度图（黑白）
- 图片最好是28x28大小，或者包含单个数字
- 背景最好是白色，数字是黑色
- 使用 `predict.py` 脚本进行预测

## 示例输出

### 训练输出示例
```
使用设备: cuda
加载数据集...
训练集大小: 60000
测试集大小: 10000

模型: simple
总参数量: 50,000

开始训练...
============================================================

Epoch [1/10]
  Batch [100/938], Loss: 0.5234, Acc: 85.23%
  ...
Train - Loss: 0.2345, Acc: 92.34%
Val   - Loss: 0.1234, Acc: 96.12%
学习率: 0.001000
✓ 保存最佳模型: ./checkpoints/simple_best.pth (Acc: 96.12%)
```

### 测试输出示例
```
总体测试准确率: 98.45% (9845/10000)

各类别准确率:
------------------------------
数字 0: 99.20% (980/987)
数字 1: 99.50% (1135/1140)
...
```

## 许可证

本项目仅供学习使用。
