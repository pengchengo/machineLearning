# encoding:utf-8
"""
MTCNN数据集处理
支持WIDER FACE数据集格式
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import glob


class FaceDataset(Dataset):
    """人脸检测数据集 - 支持WIDER FACE格式"""
    def __init__(self, data_dir, img_size=12, mode='pnet', transform=None, 
                 wider_train_dir='./wider_train', wider_annotation_file='./wider_face_split/wider_face_train_bbx_gt.txt'):
        """
        Args:
            data_dir: 数据根目录
            img_size: 图像尺寸 (12 for P-Net, 24 for R-Net, 48 for O-Net)
            mode: 网络类型 ('pnet', 'rnet', 'onet')
            transform: 数据增强
            wider_train_dir: WIDER_train图片文件夹路径（相对于data_dir或绝对路径）
            wider_annotation_file: WIDER FACE标注文件路径（相对于data_dir或绝对路径）
        """
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        
        # 构建完整路径
        self.wider_train_dir = wider_train_dir
        
        self.wider_annotation_file = wider_annotation_file
        
        # 加载数据
        self.samples = self._load_samples()
        
    def _find_image_path(self, image_name, wider_train_dir):
        """在wider_train目录及其子目录中查找图片"""
        # 支持的图片格式
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        
        # 如果image_name包含子目录路径（如 "0--Parade/0_Parade_marchingband_1_849.jpg"）
        if '/' in image_name or '\\' in image_name:
            # 直接拼接路径
            for ext in image_extensions:
                # 尝试不同的路径组合
                possible_paths = [
                    os.path.join(wider_train_dir, image_name),
                    os.path.join(wider_train_dir, image_name.replace('/', os.sep)),
                    os.path.join(wider_train_dir, image_name.replace('\\', os.sep)),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        return path
                    # 尝试添加扩展名
                    base_path = os.path.splitext(path)[0]
                    for ext_pattern in image_extensions:
                        matches = glob.glob(base_path + ext_pattern[1:])
                        if matches:
                            return matches[0]
        else:
            # 递归搜索所有子目录
            for root, dirs, files in os.walk(wider_train_dir):
                for file in files:
                    if file == image_name or os.path.splitext(file)[0] == os.path.splitext(image_name)[0]:
                        return os.path.join(root, file)
        
        return None
        
    def _load_samples(self):
        """加载WIDER FACE格式的样本数据"""
        samples = []
        
        if not os.path.exists(self.wider_annotation_file):
            print(f"错误: 未找到标注文件 {self.wider_annotation_file}")
            return samples
        
        if not os.path.exists(self.wider_train_dir):
            print(f"错误: 未找到图片目录 {self.wider_train_dir}")
            return samples
        
        print(f"正在加载WIDER FACE数据集...")
        print(f"图片目录: {self.wider_train_dir}")
        print(f"标注文件: {self.wider_annotation_file}")
        
        with open(self.wider_annotation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        valid_samples = 0
        invalid_samples = 0
        
        while i < len(lines):
            # 读取图像文件名
            image_name = lines[i].strip()
            if not image_name:
                i += 1
                continue
            
            # 查找图片路径
            img_path = self._find_image_path(image_name, self.wider_train_dir)
            
            if img_path is None or not os.path.exists(img_path):
                # 如果找不到图片，跳过这个样本
                i += 1
                # 读取边界框数量
                if i < len(lines):
                    try:
                        num_boxes = int(lines[i].strip())
                        i += num_boxes + 1  # 跳过边界框数量行和所有边界框行
                    except:
                        i += 1
                invalid_samples += 1
                continue
            
            # 读取边界框数量
            i += 1
            if i >= len(lines):
                break
            
            try:
                num_boxes = int(lines[i].strip())
            except:
                i += 1
                invalid_samples += 1
                continue
            
            i += 1
            
            # 读取边界框
            boxes = []
            for _ in range(num_boxes):
                if i >= len(lines):
                    break
                
                box_line = lines[i].strip().split()
                if len(box_line) >= 4:
                    try:
                        x1 = int(box_line[0])
                        y1 = int(box_line[1])
                        w = int(box_line[2])
                        h = int(box_line[3])
                        
                        # 检查边界框有效性
                        # WIDER FACE格式: x1 y1 w h blur expression illumination invalid occlusion pose
                        # invalid=1 表示无效边界框，应该跳过
                        invalid = int(box_line[7]) if len(box_line) > 7 else 0
                        
                        if invalid == 0 and w > 0 and h > 0:
                            # 转换为 (x1, y1, x2, y2) 格式
                            x2 = x1 + w
                            y2 = y1 + h
                            boxes.append([x1, y1, x2, y2])
                    except (ValueError, IndexError):
                        pass
                
                i += 1
            
            # 只添加有有效边界框的样本
            if boxes:
                samples.append({
                    'image_path': img_path,
                    'boxes': boxes
                })
                valid_samples += 1
        
        print(f"成功加载 {valid_samples} 个有效样本")
        if invalid_samples > 0:
            print(f"跳过 {invalid_samples} 个无效样本（图片不存在或格式错误）")
        
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
            
            # 计算偏移量（相对于裁剪区域的归一化坐标）
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w > 0 and crop_h > 0:
                # 原始边界框在裁剪区域中的相对位置
                orig_x1, orig_y1, orig_x2, orig_y2 = box
                offset_x1 = (orig_x1 - x1) / crop_w
                offset_y1 = (orig_y1 - y1) / crop_h
                offset_x2 = (orig_x2 - x1) / crop_w
                offset_y2 = (orig_y2 - y1) / crop_h
            else:
                offset_x1 = offset_y1 = offset_x2 = offset_y2 = 0.0
            
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
                     shuffle=True, num_workers=4, wider_train_dir='wider_train',
                     wider_annotation_file='wider_face_split/wider_face_train_bbx_gt.txt'):
    """
    创建数据加载器
    
    Args:
        data_dir: 数据根目录
        img_size: 图像尺寸
        mode: 网络类型
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        wider_train_dir: WIDER_train图片文件夹路径（相对于data_dir或绝对路径）
        wider_annotation_file: WIDER FACE标注文件路径（相对于data_dir或绝对路径）
    """
    dataset = FaceDataset(
        data_dir, 
        img_size=img_size, 
        mode=mode,
        wider_train_dir=wider_train_dir,
        wider_annotation_file=wider_annotation_file
    )
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
    print("WIDER FACE数据集测试")
    print("请确保数据目录结构如下:")
    print("data_dir/")
    print("  wider_train/")
    print("    0--Parade/")
    print("      0_Parade_marchingband_1_849.jpg")
    print("      ...")
    print("    1--Handshaking/")
    print("      ...")
    print("  wider_face_split/")
    print("    wider_face_train_bbx_gt.txt")
    print("\nWIDER FACE标注文件格式:")
    print("0--Parade/0_Parade_marchingband_1_849.jpg")
    print("2")
    print("449 330 122 149 0 0 0 0 0 0")
    print("361 177 92 101 0 0 0 0 0 0")
