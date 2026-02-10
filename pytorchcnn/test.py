# encoding:utf-8
"""
PyTorch CNN测试脚本 - MNIST手写数字识别
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from model import get_model
import os


def test(model, test_loader, device):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 统计每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    # 总体准确率
    accuracy = 100. * correct / total
    print(f'\n总体测试准确率: {accuracy:.2f}% ({correct}/{total})')
    
    # 每个类别的准确率
    print('\n各类别准确率:')
    print('-' * 30)
    for i in range(10):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f'数字 {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})')
        else:
            print(f'数字 {i}: N/A (0/{class_total[i]})')
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST测试')
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'advanced'],
                       help='模型类型: simple 或 advanced (默认: simple)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小 (默认: 64)')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录 (默认: ./data)')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据预处理
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载测试数据
    print('加载测试数据集...')
    test_dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform_test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f'测试集大小: {len(test_dataset)}')
    
    # 创建模型
    model = get_model(model_name=args.model, num_classes=10)
    model = model.to(device)
    
    # 加载检查点
    if not os.path.isfile(args.checkpoint):
        print(f'错误: 检查点文件不存在: {args.checkpoint}')
        return
    
    print(f'加载模型: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if 'accuracy' in checkpoint:
        print(f'训练时记录的准确率: {checkpoint["accuracy"]:.2f}%')
    
    # 测试
    print('\n开始测试...')
    print('=' * 60)
    accuracy = test(model, test_loader, device)
    print('=' * 60)
    print(f'\n测试完成！准确率: {accuracy:.2f}%')


if __name__ == '__main__':
    main()
