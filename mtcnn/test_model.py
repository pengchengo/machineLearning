# encoding:utf-8
"""
测试模型输出，诊断检测问题
"""
import torch
import cv2
import numpy as np
from mtcnn_model import get_model
import os

def test_pnet_output():
    """测试P-Net在不同输入尺寸下的输出"""
    print("=" * 60)
    print("测试P-Net输出")
    print("=" * 60)
    
    pnet = get_model('pnet')
    pnet.eval()
    
    # 测试不同输入尺寸
    test_sizes = [(12, 12), (24, 24), (48, 48), (96, 96)]
    
    for h, w in test_sizes:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            cls, bbox = pnet(x)
        print(f"输入: {w}x{h}")
        print(f"  输出形状: cls={cls.shape}, bbox={bbox.shape}")
        print(f"  分类输出范围: [{cls[0, 1].min():.3f}, {cls[0, 1].max():.3f}]")
        print()

def test_model_on_image(model_path, image_path):
    """在真实图像上测试模型"""
    print("=" * 60)
    print(f"测试模型: {model_path}")
    print(f"图像: {image_path}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return
    
    # 加载模型
    model = get_model('pnet')
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
    for scale in scales:
        hs = int(h * scale)
        ws = int(w * scale)
        if hs < 12 or ws < 12:
            continue
        
        img_resized = cv2.resize(img_rgb, (ws, hs))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        
        with torch.no_grad():
            cls_pred, bbox_pred = model(img_tensor)
        
        cls_pred = cls_pred[0, 1, :, :].numpy()
        feat_h, feat_w = cls_pred.shape
        
        print(f"\nScale: {scale:.2f}, Resized: {ws}x{hs}")
        print(f"  特征图尺寸: {feat_w}x{feat_h}")
        print(f"  分类输出范围: [{cls_pred.min():.3f}, {cls_pred.max():.3f}]")
        print(f"  平均置信度: {cls_pred.mean():.3f}")
        print(f"  大于0.5的位置数: {(cls_pred > 0.5).sum()}")
        print(f"  大于0.7的位置数: {(cls_pred > 0.7).sum()}")
        print(f"  大于0.9的位置数: {(cls_pred > 0.9).sum()}")

if __name__ == '__main__':
    import sys
    
    # 测试P-Net输出
    test_pnet_output()
    
    # 如果提供了参数，测试真实图像
    if len(sys.argv) >= 3:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        test_model_on_image(model_path, image_path)
    else:
        print("\n使用方法:")
        print("python test_model.py <模型路径> <图像路径>")
        print("例如: python test_model.py checkpoints/pnet_best.pth ./pengcheng_5.jpg")
