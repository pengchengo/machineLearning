# encoding:utf-8
"""
MTCNN人脸检测推理脚本
使用训练好的模型进行人脸检测
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
from mtcnn_model import get_model
import argparse
from PIL import Image
import time


class MTCNNDetector:
    """MTCNN检测器"""
    def __init__(self, pnet_path=None, rnet_path=None, onet_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        if pnet_path and os.path.exists(pnet_path):
            self.pnet = get_model('pnet').to(self.device)
            checkpoint = torch.load(pnet_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.pnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.pnet.load_state_dict(checkpoint)
            self.pnet.eval()
            print(f"加载P-Net: {pnet_path}")
        else:
            self.pnet = None
            print("警告: 未提供P-Net模型路径")
        
        if rnet_path and os.path.exists(rnet_path):
            self.rnet = get_model('rnet').to(self.device)
            checkpoint = torch.load(rnet_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.rnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.rnet.load_state_dict(checkpoint)
            self.rnet.eval()
            print(f"加载R-Net: {rnet_path}")
        else:
            self.rnet = None
        
        if onet_path and os.path.exists(onet_path):
            self.onet = get_model('onet').to(self.device)
            checkpoint = torch.load(onet_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.onet.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.onet.load_state_dict(checkpoint)
            self.onet.eval()
            print(f"加载O-Net: {onet_path}")
        else:
            self.onet = None
    
    def detect_pnet(self, img, min_face_size=20, scale_factor=0.709, threshold=0.6):
        """P-Net检测"""
        if self.pnet is None:
            return []
        
        h, w = img.shape[:2]
        scales = []
        m = min_face_size / 12.0
        min_length = min(h, w)
        while min_length * m >= 12:
            scales.append(m)
            m *= scale_factor
        
        boxes = []
        for scale in scales:
            hs = int(h * scale)
            ws = int(w * scale)
            if hs < 12 or ws < 12:
                continue
            
            img_resized = cv2.resize(img, (ws, hs))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
            img_tensor = img_tensor.unsqueeze(0).to(self.device) / 255.0
            
            with torch.no_grad():
                cls_pred, bbox_pred = self.pnet(img_tensor)
            
            cls_pred = cls_pred[0, 1, :, :].cpu().numpy()
            bbox_pred = bbox_pred[0].cpu().numpy()
            
            # 找到置信度高的位置
            indices = np.where(cls_pred > threshold)
            if len(indices[0]) == 0:
                continue
            
            for i, j in zip(indices[0], indices[1]):
                score = cls_pred[i, j]
                offset = bbox_pred[:, i, j]
                
                # 计算原始坐标
                x1 = int((j * 2) / scale)
                y1 = int((i * 2) / scale)
                x2 = int((j * 2 + 12) / scale)
                y2 = int((i * 2 + 12) / scale)
                
                # 应用偏移
                x1 = int(x1 + offset[0] * 12 / scale)
                y1 = int(y1 + offset[1] * 12 / scale)
                x2 = int(x2 + offset[2] * 12 / scale)
                y2 = int(y2 + offset[3] * 12 / scale)
                
                boxes.append([x1, y1, x2, y2, score])
        
        return self.nms(boxes, 0.5)
    
    def detect_rnet(self, img, boxes):
        """R-Net精炼"""
        if self.rnet is None or len(boxes) == 0:
            return boxes
        
        refined_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            w = x2 - x1
            h = y2 - y1
            
            # 扩展边界框
            x1 = max(0, int(x1 - 0.1 * w))
            y1 = max(0, int(y1 - 0.1 * h))
            x2 = min(img.shape[1], int(x2 + 0.1 * w))
            y2 = min(img.shape[0], int(y2 + 0.1 * h))
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop = cv2.resize(crop, (24, 24))
            crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
            crop_tensor = crop_tensor.unsqueeze(0).to(self.device) / 255.0
            
            with torch.no_grad():
                cls_pred, bbox_pred = self.rnet(crop_tensor)
            
            score = cls_pred[0, 1].item()
            if score > 0.7:
                offset = bbox_pred[0].cpu().numpy()
                x1 = int(x1 + offset[0] * w)
                y1 = int(y1 + offset[1] * h)
                x2 = int(x2 + offset[2] * w)
                y2 = int(y2 + offset[3] * h)
                refined_boxes.append([x1, y1, x2, y2, score])
        
        return self.nms(refined_boxes, 0.5)
    
    def detect_onet(self, img, boxes):
        """O-Net最终输出"""
        if self.onet is None or len(boxes) == 0:
            return boxes
        
        final_boxes = []
        landmarks_list = []
        
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            w = x2 - x1
            h = y2 - y1
            
            # 扩展边界框
            x1 = max(0, int(x1 - 0.2 * w))
            y1 = max(0, int(y1 - 0.2 * h))
            x2 = min(img.shape[1], int(x2 + 0.2 * w))
            y2 = min(img.shape[0], int(y2 + 0.2 * h))
            
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop = cv2.resize(crop, (48, 48))
            crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float()
            crop_tensor = crop_tensor.unsqueeze(0).to(self.device) / 255.0
            
            with torch.no_grad():
                cls_pred, bbox_pred, landmarks_pred = self.onet(crop_tensor)
            
            score = cls_pred[0, 1].item()
            if score > 0.8:
                offset = bbox_pred[0].cpu().numpy()
                landmarks = landmarks_pred[0].cpu().numpy()
                
                x1 = int(x1 + offset[0] * w)
                y1 = int(y1 + offset[1] * h)
                x2 = int(x2 + offset[2] * w)
                y2 = int(y2 + offset[3] * h)
                
                # 关键点坐标
                landmarks = landmarks.reshape(5, 2)
                landmarks[:, 0] = landmarks[:, 0] * w + x1
                landmarks[:, 1] = landmarks[:, 1] * h + y1
                
                final_boxes.append([x1, y1, x2, y2, score])
                landmarks_list.append(landmarks)
        
        return final_boxes, landmarks_list
    
    def nms(self, boxes, threshold):
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return boxes[keep].tolist()
    
    def detect(self, img):
        """完整检测流程"""
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"无法读取图像: {img}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # P-Net
        boxes = self.detect_pnet(img_rgb)
        
        # R-Net
        if self.rnet is not None:
            boxes = self.detect_rnet(img_rgb, boxes)
        
        # O-Net
        if self.onet is not None:
            boxes, landmarks = self.detect_onet(img_rgb, boxes)
            return boxes, landmarks
        
        return boxes, []


def visualize_detection(img, boxes, landmarks_list=None):
    """可视化检测结果"""
    img_vis = img.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        score = box[4] if len(box) > 4 else 1.0
        
        # 绘制边界框
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制置信度
        cv2.putText(img_vis, f'{score:.2f}', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制关键点
        if landmarks_list and i < len(landmarks_list):
            landmarks = landmarks_list[i]
            for point in landmarks:
                cv2.circle(img_vis, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)
    
    return img_vis


def main():
    parser = argparse.ArgumentParser(description='MTCNN人脸检测')
    parser.add_argument('--image', type=str, required=True,
                       help='输入图像路径')
    parser.add_argument('--pnet', type=str, default=None,
                       help='P-Net模型路径')
    parser.add_argument('--rnet', type=str, default=None,
                       help='R-Net模型路径')
    parser.add_argument('--onet', type=str, default=None,
                       help='O-Net模型路径')
    parser.add_argument('--output', type=str, default='result.jpg',
                       help='输出图像路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='使用设备')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = MTCNNDetector(
        pnet_path=args.pnet,
        rnet_path=args.rnet,
        onet_path=args.onet,
        device=args.device
    )
    
    # 读取图像
    img = cv2.imread(args.image)
    if img is None:
        print(f"错误: 无法读取图像 {args.image}")
        return
    
    # 检测
    print("开始检测...")
    start_time = time.time()
    boxes, landmarks = detector.detect(img)
    elapsed_time = time.time() - start_time
    
    print(f"检测到 {len(boxes)} 个人脸，耗时: {elapsed_time:.3f}秒")
    
    # 可视化
    img_result = visualize_detection(img, boxes, landmarks)
    
    # 保存结果
    cv2.imwrite(args.output, img_result)
    print(f"结果已保存到: {args.output}")
    
    # 显示结果
    cv2.imshow('Detection Result', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
#python detect_mtcnn.py --image ./pengcheng_5.jpg --pnet checkpoints/pnet_best.pth --rnet checkpoints/rnet_best.pth --onet checkpoints/onet_best.pth --output result.jpg --device cuda
