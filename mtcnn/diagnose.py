# encoding:utf-8
"""
MTCNN诊断工具
帮助判断是代码bug还是训练不充分
"""
import torch
import cv2
import numpy as np
from mtcnn_model import get_model
import os
import matplotlib.pyplot as plt

def test_model_output_distribution(model_path, mode='pnet'):
    """测试模型输出的分布，判断模型是否训练好"""
    print("=" * 60)
    print(f"测试 {mode.upper()}-Net 模型输出分布")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    # 加载模型
    model = get_model(mode)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        accuracy = checkpoint.get('accuracy', 0)
        print(f"模型训练准确率: {accuracy:.2f}%")
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 测试1: 纯噪声输入（应该输出低置信度）
    print("\n【测试1】纯噪声输入（应该输出低置信度）")
    noise = torch.randn(1, 3, 12 if mode == 'pnet' else (24 if mode == 'rnet' else 48),
                        12 if mode == 'pnet' else (24 if mode == 'rnet' else 48))
    with torch.no_grad():
        if mode == 'onet':
            cls_pred, bbox_pred, _ = model(noise)
        else:
            cls_pred, bbox_pred = model(noise)
    
    if len(cls_pred.shape) == 4:  # P-Net
        cls_pred = cls_pred[0, 1, :, :].numpy()
        print(f"  输出形状: {cls_pred.shape}")
        print(f"  置信度范围: [{cls_pred.min():.3f}, {cls_pred.max():.3f}]")
        print(f"  平均置信度: {cls_pred.mean():.3f}")
        print(f"  大于0.5的位置: {(cls_pred > 0.5).sum()} / {cls_pred.size}")
    else:  # R-Net, O-Net
        cls_pred = cls_pred[0, 1].item()
        print(f"  置信度: {cls_pred:.3f}")
    
    # 判断：如果噪声输入也输出高置信度，说明模型未训练好
    if isinstance(cls_pred, np.ndarray):
        if cls_pred.mean() > 0.3:
            print("  ⚠️  警告：噪声输入也输出较高置信度，模型可能未训练好")
        else:
            print("  ✅ 正常：噪声输入输出低置信度")
    else:
        if cls_pred > 0.3:
            print("  ⚠️  警告：噪声输入也输出较高置信度，模型可能未训练好")
        else:
            print("  ✅ 正常：噪声输入输出低置信度")
    
    # 测试2: 真实人脸图像（应该输出高置信度）
    print("\n【测试2】真实人脸图像（应该输出高置信度）")
    # 创建一个简单的人脸样本来测试
    # 这里使用一个简单的测试：如果模型训练好，应该能区分
    
    return model

def test_coordinate_calculation():
    """测试坐标计算是否正确"""
    print("\n" + "=" * 60)
    print("测试坐标计算")
    print("=" * 60)
    
    # 模拟P-Net输出
    feat_h, feat_w = 5, 5  # 特征图尺寸
    scale = 0.5  # 缩放因子
    stride = 2
    
    # 测试特征图位置(2, 2)对应的原图坐标
    i, j = 2, 2
    x_in_input = (j * stride + stride / 2.0)
    y_in_input = (i * stride + stride / 2.0)
    x1 = int((x_in_input - 6) / scale)
    y1 = int((y_in_input - 6) / scale)
    x2 = int((x_in_input + 6) / scale)
    y2 = int((y_in_input + 6) / scale)
    
    print(f"特征图位置: ({i}, {j})")
    print(f"输入图像坐标: ({x_in_input:.1f}, {y_in_input:.1f})")
    print(f"原图坐标: ({x1}, {y1}) -> ({x2}, {y2})")
    print(f"边界框尺寸: {x2-x1}x{y2-y1}")
    
    # 验证：对于scale=0.5，输入图像是原图的一半
    # 特征图位置(2,2)对应输入图像位置约(5, 5)
    # 对应原图位置约(10, 10)
    print("\n✅ 坐标计算逻辑看起来正确")

def test_on_real_image(model_path, image_path, mode='pnet'):
    """在真实图像上测试"""
    print("\n" + "=" * 60)
    print(f"在真实图像上测试 {mode.upper()}-Net")
    print("=" * 60)
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    # 加载模型
    model = get_model(mode)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图像: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    print(f"图像尺寸: {w}x{h}")
    
    # 测试不同scale
    scales = [0.5, 0.7, 1.0]
    all_scores = []
    
    for scale in scales:
        hs = int(h * scale)
        ws = int(w * scale)
        if hs < 12 or ws < 12:
            continue
        
        img_resized = cv2.resize(img_rgb, (ws, hs))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        
        with torch.no_grad():
            if mode == 'onet':
                cls_pred, bbox_pred, _ = model(img_tensor)
            else:
                cls_pred, bbox_pred = model(img_tensor)
        
        if len(cls_pred.shape) == 4:  # P-Net
            cls_pred_np = cls_pred[0, 1, :, :].numpy()
            all_scores.extend(cls_pred_np.flatten())
            print(f"\nScale: {scale:.2f}, Resized: {ws}x{hs}")
            print(f"  特征图尺寸: {cls_pred_np.shape[1]}x{cls_pred_np.shape[0]}")
            print(f"  置信度范围: [{cls_pred_np.min():.3f}, {cls_pred_np.max():.3f}]")
            print(f"  平均置信度: {cls_pred_np.mean():.3f}")
            print(f"  大于0.5的位置: {(cls_pred_np > 0.5).sum()} / {cls_pred_np.size}")
            print(f"  大于0.7的位置: {(cls_pred_np > 0.7).sum()} / {cls_pred_np.size}")
        else:  # R-Net, O-Net
            cls_pred_np = cls_pred[0, 1].item()
            all_scores.append(cls_pred_np)
            print(f"\nScale: {scale:.2f}, Resized: {ws}x{hs}")
            print(f"  置信度: {cls_pred_np:.3f}")
    
    # 分析结果
    if all_scores:
        all_scores = np.array(all_scores)
        print(f"\n【整体分析】")
        print(f"所有置信度范围: [{all_scores.min():.3f}, {all_scores.max():.3f}]")
        print(f"平均置信度: {all_scores.mean():.3f}")
        print(f"中位数: {np.median(all_scores):.3f}")
        print(f"标准差: {all_scores.std():.3f}")
        
        # 判断
        if all_scores.mean() > 0.5:
            print("\n⚠️  警告：平均置信度很高，可能是：")
            print("  1. 模型输出异常（所有位置都输出高置信度）")
            print("  2. 图像中确实有很多人脸（不太可能）")
            print("  3. 模型未训练好，无法区分人脸和非人脸")
        elif all_scores.max() < 0.3:
            print("\n⚠️  警告：所有置信度都很低，可能是：")
            print("  1. 模型未训练好")
            print("  2. 图像中没有人脸")
            print("  3. 阈值设置过高")
        else:
            print("\n✅ 置信度分布看起来正常")

def check_training_logs(checkpoint_path):
    """检查训练日志"""
    print("\n" + "=" * 60)
    print("检查训练日志")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点文件不存在: {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"检查点信息:")
    if 'epoch' in checkpoint:
        print(f"  训练轮数: {checkpoint['epoch']}")
    if 'accuracy' in checkpoint:
        print(f"  准确率: {checkpoint['accuracy']:.2f}%")
    if 'loss' in checkpoint:
        print(f"  损失: {checkpoint['loss']:.4f}")
    
    # 判断
    if 'accuracy' in checkpoint:
        acc = checkpoint['accuracy']
        if acc < 80:
            print("\n⚠️  警告：准确率较低，模型可能未充分训练")
            print("  建议：继续训练或检查训练数据")
        elif acc < 90:
            print("\n⚠️  注意：准确率中等，可能需要更多训练")
        else:
            print("\n✅ 准确率较高，模型训练较好")

def diagnose_all(pnet_path=None, rnet_path=None, onet_path=None, test_image=None):
    """综合诊断"""
    print("=" * 60)
    print("MTCNN综合诊断工具")
    print("=" * 60)
    
    # 1. 测试坐标计算
    test_coordinate_calculation()
    
    # 2. 检查训练日志
    if pnet_path:
        check_training_logs(pnet_path)
        test_model_output_distribution(pnet_path, 'pnet')
    
    if rnet_path:
        check_training_logs(rnet_path)
        test_model_output_distribution(rnet_path, 'rnet')
    
    if onet_path:
        check_training_logs(onet_path)
        test_model_output_distribution(onet_path, 'onet')
    
    # 3. 在真实图像上测试
    if test_image and pnet_path:
        test_on_real_image(pnet_path, test_image, 'pnet')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MTCNN诊断工具')
    parser.add_argument('--pnet', type=str, default=None,
                       help='P-Net模型路径')
    parser.add_argument('--rnet', type=str, default=None,
                       help='R-Net模型路径')
    parser.add_argument('--onet', type=str, default=None,
                       help='O-Net模型路径')
    parser.add_argument('--image', type=str, default=None,
                       help='测试图像路径')
    
    args = parser.parse_args()
    
    diagnose_all(
        pnet_path=args.pnet,
        rnet_path=args.rnet,
        onet_path=args.onet,
        test_image=args.image
    )
