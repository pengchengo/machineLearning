# encoding:utf-8
"""
PyTorch CNN预测脚本 - 单张图片预测
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import numpy as np
from model import get_model
import os


def predict_image(model, image_path, device, transform):
    """预测单张图片"""
    # 加载图片
    try:
        image = Image.open(image_path).convert('L')  # 转为灰度图
    except Exception as e:
        print(f'错误: 无法加载图片 {image_path}: {e}')
        return None
    
    # 预处理
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    image_tensor = image_tensor.to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)
        predicted = predicted.item()
        confidence = confidence.item()
    
    # 获取所有类别的概率
    probs = probabilities[0].cpu().numpy()
    
    return predicted, confidence, probs


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST单张图片预测')
    parser.add_argument('--model', type=str, default='simple',
                       choices=['simple', 'advanced'],
                       help='模型类型: simple 或 advanced (默认: simple)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True,
                       help='要预测的图片路径')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据预处理（与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整大小
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
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
    
    # 预测
    print(f'\n预测图片: {args.image}')
    print('=' * 60)
    
    result = predict_image(model, args.image, device, transform)
    
    if result is None:
        return
    
    predicted, confidence, probs = result
    
    print(f'\n预测结果: 数字 {predicted}')
    print(f'置信度: {confidence*100:.2f}%')
    print('\n所有类别的概率:')
    print('-' * 30)
    for i in range(10):
        print(f'数字 {i}: {probs[i]*100:6.2f}%')
    
    print('=' * 60)


if __name__ == '__main__':
    main()
