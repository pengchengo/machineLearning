# encoding:utf-8
"""
MTCNN数据集处理
支持WIDER FACE等格式
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


class FaceDataset(Dataset):
    """人脸检测数据集"""
    def __init__(self, data_dir, img_size=12, mode='pnet', transform=None):
        """
        Args:
            data_dir: 数据目录，包含images和annotations
            img_size: 图像尺寸 (12 for P-Net, 24 for R-Net, 48 for O-Net)
            mode: 网络类型 ('pnet', 'rnet', 'onet')
            transform: 数据增强
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        
        # 加载数据
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """加载样本数据"""
        samples = []
        images_dir = os.path.join(self.data_dir, 'images')
        annotations_file = os.path.join(self.data_dir, 'annotations.txt')
        
        if not os.path.exists(annotations_file):
            print(f"警告: 未找到标注文件 {annotations_file}")
            print("请创建标注文件，格式: image_path x1 y1 x2 y2 (每行一个边界框)")
            return samples
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                
                img_path = parts[0]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(images_dir, img_path)
                
                # 读取边界框坐标
                boxes = []
                for i in range(1, len(parts), 4):
                    if i + 3 < len(parts):
                        x1 = int(parts[i])
                        y1 = int(parts[i+1])
                        x2 = int(parts[i+2])
                        y2 = int(parts[i+3])
                        boxes.append([x1, y1, x2, y2])
                
                if boxes and os.path.exists(img_path):
                    samples.append({
                        'image_path': img_path,
                        'boxes': boxes
                    })
        
        print(f"加载了 {len(samples)} 个样本")
        return samples
    
    def __len__(self):
        return len(self.samples) * 10  # 数据增强，每个样本生成多个训练样本
    
    def __getitem__(self, idx):
        # 随机选择一个样本
        sample_idx = idx % len(self.samples)
        sample = self.samples[sample_idx]
        
        # 读取图像
        img = cv2.imread(sample['image_path'])
        if img is None:
            # 如果图像读取失败，返回一个随机样本
            return self.__getitem__((idx + 1) % len(self.samples))
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 随机选择一个边界框或生成负样本
        if random.random() < 0.5 and sample['boxes']:
            # 正样本：从真实边界框生成
            box = random.choice(sample['boxes'])
            x1, y1, x2, y2 = box
            
            # 扩展边界框
            width = x2 - x1
            height = y2 - y1
            size = max(width, height)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 随机扩展
            size = int(size * random.uniform(1.2, 1.5))
            x1 = max(0, int(center_x - size / 2))
            y1 = max(0, int(center_y - size / 2))
            x2 = min(w, int(center_x + size / 2))
            y2 = min(h, int(center_y + size / 2))
            
            # 裁剪并调整大小
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                return self.__getitem__((idx + 1) % len(self.samples))
            
            crop = cv2.resize(crop, (self.img_size, self.img_size))
            
            # 计算偏移量
            offset_x1 = (sample['boxes'][0][0] - x1) / (x2 - x1)
            offset_y1 = (sample['boxes'][0][1] - y1) / (y2 - y1)
            offset_x2 = (sample['boxes'][0][2] - x1) / (x2 - x1)
            offset_y2 = (sample['boxes'][0][3] - y1) / (y2 - y1)
            
            label = 1  # 正样本
            bbox_target = np.array([offset_x1, offset_y1, offset_x2, offset_y2], dtype=np.float32)
            
        else:
            # 负样本：随机裁剪
            size = random.randint(self.img_size, min(w, h) // 2)
            x1 = random.randint(0, max(1, w - size))
            y1 = random.randint(0, max(1, h - size))
            x2 = x1 + size
            y2 = y1 + size
            
            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (self.img_size, self.img_size))
            
            label = 0  # 负样本
            bbox_target = np.array([0, 0, 0, 0], dtype=np.float32)
        
        # 转换为tensor
        crop = crop.astype(np.float32) / 255.0
        crop = np.transpose(crop, (2, 0, 1))  # HWC -> CHW
        
        if self.transform:
            crop = self.transform(torch.from_numpy(crop))
        else:
            crop = torch.from_numpy(crop)
        
        # 分类标签
        cls_label = torch.tensor(label, dtype=torch.long)
        
        # 边界框回归标签
        bbox_label = torch.from_numpy(bbox_target)
        
        if self.mode == 'onet':
            # O-Net还需要关键点标签（这里简化处理，实际需要关键点标注）
            landmarks_label = torch.zeros(10, dtype=torch.float32)
            return crop, cls_label, bbox_label, landmarks_label
        
        return crop, cls_label, bbox_label


def create_dataloader(data_dir, img_size=12, mode='pnet', batch_size=32, 
                     shuffle=True, num_workers=4):
    """创建数据加载器"""
    dataset = FaceDataset(data_dir, img_size=img_size, mode=mode)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    # 测试数据集
    # 注意：需要先创建数据目录和标注文件
    print("数据集测试")
    print("请确保数据目录结构如下:")
    print("data_dir/")
    print("  images/")
    print("    img1.jpg")
    print("    img2.jpg")
    print("  annotations.txt")
    print("\nannotations.txt格式:")
    print("img1.jpg x1 y1 x2 y2")
    print("img2.jpg x1 y1 x2 y2 x1 y1 x2 y2")
