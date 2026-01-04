# MTCNN人脸检测 - PyTorch实现

基于PyTorch实现的MTCNN（Multi-task CNN）人脸检测算法，支持在RTX 5070 Ti等GPU上进行训练和推理。

## 算法特点

- **三阶段检测**：P-Net（快速候选）→ R-Net（精炼）→ O-Net（最终输出）
- **多任务学习**：同时进行人脸分类、边界框回归和关键点检测
- **高精度**：相比传统方法（如Haar Cascade）精度显著提升
- **GPU加速**：充分利用RTX 5070 Ti的CUDA加速能力

## 项目结构

```
faceRecognition/
├── mtcnn_model.py          # 模型定义（P-Net, R-Net, O-Net）
├── mtcnn_dataset.py        # 数据集处理
├── train_mtcnn.py          # 训练脚本
├── detect_mtcnn.py         # 检测/推理脚本
├── requirements_mtcnn.txt  # 依赖包
└── README_mtcnn.md         # 说明文档
```

## 安装依赖

```bash
pip install -r requirements_mtcnn.txt
```

## 数据准备

### 数据目录结构

```
data/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── annotations.txt
```

### 标注文件格式

`annotations.txt` 格式：每行一个图像，包含图像路径和边界框坐标

```
images/img1.jpg x1 y1 x2 y2
images/img2.jpg x1 y1 x2 y2 x1 y1 x2 y2
```

**注意**：
- 坐标格式：`x1 y1 x2 y2`（左上角和右下角）
- 一张图像可以有多个边界框
- 图像路径可以是相对路径（相对于images目录）或绝对路径

### 推荐数据集

1. **WIDER FACE**：大规模人脸检测数据集
   - 下载：http://shuoyang1213.me/WIDERFACE/
   - 需要转换为上述格式

2. **FDDB**：另一个常用的人脸检测数据集

3. **自定义数据**：使用自己的图像和标注

## 训练

### 训练P-Net

```bash
python train_mtcnn.py \
    --mode pnet \
    --data_dir ./data \
    --batch_size 64 \
    --epochs 20 \
    --lr 0.001 \
    --save_dir ./checkpoints
```

### 训练R-Net

```bash
python train_mtcnn.py \
    --mode rnet \
    --data_dir ./data \
    --batch_size 32 \
    --epochs 20 \
    --lr 0.001 \
    --save_dir ./checkpoints
```

### 训练O-Net

```bash
python train_mtcnn.py \
    --mode onet \
    --data_dir ./data \
    --batch_size 16 \
    --epochs 20 \
    --lr 0.001 \
    --save_dir ./checkpoints
```

### 训练参数说明

- `--mode`: 网络类型（pnet/rnet/onet）
- `--data_dir`: 数据目录路径
- `--batch_size`: 批次大小（根据GPU内存调整）
- `--epochs`: 训练轮数
- `--lr`: 学习率
- `--save_dir`: 模型保存目录
- `--resume`: 恢复训练的检查点路径（可选）

### 训练建议

1. **批次大小**：RTX 5070 Ti建议
   - P-Net: 64-128
   - R-Net: 32-64
   - O-Net: 16-32

2. **学习率**：初始0.001，每10个epoch降低10倍

3. **训练顺序**：先训练P-Net，再训练R-Net，最后训练O-Net

4. **监控训练**：使用TensorBoard查看训练过程
   ```bash
   tensorboard --logdir ./runs
   ```

## 推理/检测

### 基本使用

```bash
python detect_mtcnn.py \
    --image ./find/ping3.jpg \
    --pnet ./checkpoints/pnet_best.pth \
    --rnet ./checkpoints/rnet_best.pth \
    --onet ./checkpoints/onet_best.pth \
    --output result.jpg
```

### 仅使用P-Net（快速检测）

```bash
python detect_mtcnn.py \
    --image ./find/ping3.jpg \
    --pnet ./checkpoints/pnet_best.pth \
    --output result.jpg
```

### Python代码使用

```python
from detect_mtcnn import MTCNNDetector
import cv2

# 创建检测器
detector = MTCNNDetector(
    pnet_path='./checkpoints/pnet_best.pth',
    rnet_path='./checkpoints/rnet_best.pth',
    onet_path='./checkpoints/onet_best.pth',
    device='cuda'
)

# 检测
img = cv2.imread('test.jpg')
boxes, landmarks = detector.detect(img)

# 绘制结果
for box in boxes:
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('result', img)
cv2.waitKey(0)
```

## 性能优化

### RTX 5070 Ti优化建议

1. **混合精度训练**：使用`torch.cuda.amp`加速训练
2. **数据加载**：设置`num_workers=4-8`，使用`pin_memory=True`
3. **批次大小**：根据GPU内存最大化批次大小
4. **推理优化**：使用`torch.jit.script`或`torch.jit.trace`导出优化模型

## 常见问题

### 1. 内存不足（OOM）

- 减小`batch_size`
- 减小图像尺寸
- 使用梯度累积

### 2. 检测精度低

- 增加训练数据
- 调整检测阈值（`threshold`参数）
- 使用数据增强

### 3. 训练速度慢

- 检查GPU使用率（`nvidia-smi`）
- 增加`num_workers`
- 使用混合精度训练

## 与现有代码对比

相比你现有的`faceDetection.py`（Haar Cascade）：
- ✅ 精度更高
- ✅ 可训练和微调
- ✅ 支持关键点检测
- ✅ GPU加速
- ⚠️ 需要训练数据
- ⚠️ 推理速度稍慢（但GPU上仍然很快）

## 下一步

1. **数据准备**：准备或下载训练数据
2. **训练模型**：按顺序训练P-Net、R-Net、O-Net
3. **评估测试**：在测试集上评估性能
4. **优化部署**：针对实际应用场景优化

## 参考资源

- MTCNN原论文：Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
- PyTorch官方文档：https://pytorch.org/
- WIDER FACE数据集：http://shuoyang1213.me/WIDERFACE/

## 许可证

本项目仅供学习和研究使用。
