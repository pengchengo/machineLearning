# encoding:utf-8
"""
快速开始示例 - 演示如何使用PyTorch CNN进行MNIST手写数字识别
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model


def quick_train():
    """快速训练示例"""
    print("=" * 60)
    print("PyTorch CNN MNIST 快速训练示例")
    print("=" * 60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}\n')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据（只使用部分数据加快速度）
    print('加载数据集...')
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 为了快速演示，只使用前5000张图片
    train_dataset = torch.utils.data.Subset(train_dataset, range(5000))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False
    )
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'测试集大小: {len(test_dataset)}\n')
    
    # 创建模型
    model = get_model('simple', num_classes=10)
    model = model.to(device)
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}\n')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练3个epoch（快速演示）
    print('开始训练（3个epoch，快速演示）...')
    print('-' * 60)
    
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        print(f'Epoch [{epoch+1}/3] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    # 测试
    print('\n开始测试...')
    print('-' * 60)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'测试准确率: {accuracy:.2f}% ({correct}/{total})')
    print('=' * 60)
    print('\n快速训练完成！')
    print('提示: 使用 train.py 进行完整训练可以获得更好的效果')
    print('命令: python train.py --model simple --epochs 10')


if __name__ == '__main__':
    quick_train()
