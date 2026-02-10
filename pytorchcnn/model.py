# encoding:utf-8
"""
PyTorch CNN模型定义 - 手写数字识别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """简单的CNN模型用于MNIST手写数字识别"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 第一个卷积块: 28x28x1 -> 26x26x8 -> 13x13x8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块: 13x13x8 -> 11x11x16 -> 5x5x16
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层: 5*5*16 = 400 -> 128 -> 10
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.conv1(x))  # (batch, 1, 28, 28) -> (batch, 8, 26, 26)
        x = self.pool1(x)          # (batch, 8, 26, 26) -> (batch, 8, 13, 13)
        
        # 第二个卷积块
        x = F.relu(self.conv2(x))  # (batch, 8, 13, 13) -> (batch, 16, 11, 11)
        x = self.pool2(x)          # (batch, 16, 11, 11) -> (batch, 16, 5, 5)
        
        # 展平
        x = x.view(x.size(0), -1)  # (batch, 16, 5, 5) -> (batch, 400)
        
        # 全连接层
        x = F.relu(self.fc1(x))    # (batch, 400) -> (batch, 128)
        x = self.dropout(x)
        x = self.fc2(x)            # (batch, 128) -> (batch, 10)
        
        return x


class AdvancedCNN(nn.Module):
    """更高级的CNN模型，性能更好"""
    
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        # 第一个卷积块: 28x28x1 -> 28x28x32 -> 14x14x32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块: 14x14x32 -> 14x14x64 -> 7x7x64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块: 7x7x64 -> 7x7x128 -> 3x3x128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层: 3*3*128 = 1152 -> 256 -> 10
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 1, 28, 28) -> (batch, 32, 28, 28)
        x = self.pool1(x)                     # (batch, 32, 28, 28) -> (batch, 32, 14, 14)
        
        # 第二个卷积块
        x = F.relu(self.bn2(self.conv2(x)))   # (batch, 32, 14, 14) -> (batch, 64, 14, 14)
        x = self.pool2(x)                     # (batch, 64, 14, 14) -> (batch, 64, 7, 7)
        
        # 第三个卷积块
        x = F.relu(self.bn3(self.conv3(x)))   # (batch, 64, 7, 7) -> (batch, 128, 7, 7)
        x = self.pool3(x)                     # (batch, 128, 7, 7) -> (batch, 128, 3, 3)
        
        # 展平
        x = x.view(x.size(0), -1)             # (batch, 128, 3, 3) -> (batch, 1152)
        
        # 全连接层
        x = F.relu(self.fc1(x))               # (batch, 1152) -> (batch, 256)
        x = self.dropout(x)
        x = self.fc2(x)                       # (batch, 256) -> (batch, 10)
        
        return x


def get_model(model_name='simple', num_classes=10):
    """
    获取模型
    
    Args:
        model_name: 'simple' 或 'advanced'
        num_classes: 分类数量，MNIST为10
    
    Returns:
        模型实例
    """
    if model_name == 'simple':
        return SimpleCNN(num_classes=num_classes)
    elif model_name == 'advanced':
        return AdvancedCNN(num_classes=num_classes)
    else:
        raise ValueError(f"未知的模型名称: {model_name}，请选择 'simple' 或 'advanced'")


if __name__ == '__main__':
    # 测试模型
    print("测试SimpleCNN模型...")
    model = SimpleCNN()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n测试AdvancedCNN模型...")
    model = AdvancedCNN()
    x = torch.randn(1, 1, 28, 28)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
