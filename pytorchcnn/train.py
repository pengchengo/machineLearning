# encoding:utf-8
"""
PyTorch CNN训练脚本 - MNIST手写数字识别
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
from model import get_model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 打印进度
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST训练')
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['simple', 'advanced'],
                       help='模型类型: simple 或 advanced (默认: simple)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (默认: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数 (默认: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率 (默认: 0.001)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录 (默认: ./data)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录 (默认: ./checkpoints)')
    parser.add_argument('--log_dir', type=str, default='./runs',
                       help='TensorBoard日志目录 (默认: ./runs)')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据
    print('加载数据集...')
    train_dataset = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    # 创建模型
    model = get_model(model_name=args.model, num_classes=10)
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n模型: {args.model}')
    print(f'总参数量: {total_params:,}')
    print(f'可训练参数量: {trainable_params:,}')
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(args.log_dir, f'{args.model}_train'))
    
    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'加载检查点: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint.get('accuracy', 0.0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'从epoch {start_epoch}恢复训练，最佳准确率: {best_acc:.2f}%')
        else:
            print(f'警告: 检查点文件不存在: {args.resume}')
    
    # 训练循环
    print(f'\n开始训练...')
    print('=' * 60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'学习率: {current_lr:.6f}')
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'loss': val_loss,
            }, checkpoint_path)
            print(f'✓ 保存最佳模型: {checkpoint_path} (Acc: {val_acc:.2f}%)')
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'{args.model}_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'loss': val_loss,
            }, checkpoint_path)
            print(f'✓ 保存检查点: {checkpoint_path}')
    
    print('\n' + '=' * 60)
    print(f'训练完成！最佳验证准确率: {best_acc:.2f}%')
    print(f'模型保存在: {args.save_dir}')
    print(f'TensorBoard日志: {args.log_dir}')
    
    writer.close()


if __name__ == '__main__':
    main()
