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
    
    def detect_pnet(self, img, min_face_size=20, scale_factor=0.709, threshold=0.7):
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
            
            # 计算特征图尺寸
            feat_h, feat_w = cls_pred.shape
            
            # P-Net的stride计算
            # conv1(3x3, no padding) -> pool1(2x2, stride=2) -> conv2(3x3) -> conv3(3x3)
            # 总stride = 2 (来自pool1)
            stride = 2
            cell_size = 12  # P-Net的输入patch大小
            
            # 计算特征图到输入图像的映射
            # 每个特征图位置对应输入图像中的一个12x12区域
            # 特征图位置(i,j)对应输入图像中中心位置约为 (j*stride+stride/2, i*stride+stride/2)
            # 但需要考虑实际的网络结构，这里使用简化的映射
            
            # 找到置信度高的位置
            indices = np.where(cls_pred > threshold)
            if len(indices[0]) == 0:
                continue
            
            for i, j in zip(indices[0], indices[1]):
                score = cls_pred[i, j]
                offset = bbox_pred[:, i, j]
                
                # 计算特征图位置对应的输入图像坐标
                # 特征图位置(i,j)对应输入图像中的位置
                # 使用stride=2的映射
                x_in_input = (j * stride + stride / 2.0)
                y_in_input = (i * stride + stride / 2.0)
                
                # 初始边界框（以特征图位置为中心，大小为cell_size）
                x1_in_input = x_in_input - cell_size / 2.0
                y1_in_input = y_in_input - cell_size / 2.0
                x2_in_input = x_in_input + cell_size / 2.0
                y2_in_input = y_in_input + cell_size / 2.0
                
                # 映射回原始图像坐标
                x1 = int(x1_in_input / scale)
                y1 = int(y1_in_input / scale)
                x2 = int(x2_in_input / scale)
                y2 = int(y2_in_input / scale)
                
                # 应用偏移（offset是归一化的，相对于12x12的crop区域）
                # 训练时：offset = (orig_box - crop_box) / crop_size
                # 检测时：orig_box = crop_box + offset * crop_size
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w > 0 and box_h > 0:
                    # offset[0], offset[1]是左上角的归一化偏移
                    # offset[2], offset[3]是右下角的归一化偏移
                    x1_new = x1 + offset[0] * box_w
                    y1_new = y1 + offset[1] * box_h
                    x2_new = x1 + offset[2] * box_w
                    y2_new = y1 + offset[3] * box_h
                    
                    x1 = int(x1_new)
                    y1 = int(y1_new)
                    x2 = int(x2_new)
                    y2 = int(y2_new)
                
                # 确保坐标在图像范围内
                x1 = max(0, min(x1, w))
                y1 = max(0, min(y1, h))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                
                # 只保留有效的边界框（最小尺寸过滤）
                if x2 > x1 and y2 > y1 and (x2 - x1) > 10 and (y2 - y1) > 10:
                    boxes.append([x1, y1, x2, y2, score])
        
        return self.nms(boxes, 0.4)  # 降低NMS阈值，更严格地过滤重叠框
    
    def detect_rnet(self, img, boxes, threshold=0.8):
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
            if score > threshold:
                offset = bbox_pred[0].cpu().numpy()
                x1 = int(x1 + offset[0] * w)
                y1 = int(y1 + offset[1] * h)
                x2 = int(x2 + offset[2] * w)
                y2 = int(y2 + offset[3] * h)
                refined_boxes.append([x1, y1, x2, y2, score])
        
        return self.nms(refined_boxes, 0.4)
    
    def detect_onet(self, img, boxes, threshold=0.9):
        """O-Net最终输出"""
        if self.onet is None or len(boxes) == 0:
            return boxes, []
        
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
            if score > threshold:
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
    
    def detect(self, img, pnet_threshold=0.7, rnet_threshold=0.8, onet_threshold=0.9, nms_threshold=0.4):
        """
        完整检测流程
        
        Args:
            img: 输入图像（路径或numpy数组）
            pnet_threshold: P-Net置信度阈值（默认0.7，越高越严格）
            rnet_threshold: R-Net置信度阈值（默认0.8）
            onet_threshold: O-Net置信度阈值（默认0.9）
            nms_threshold: NMS IoU阈值（默认0.4，越低越严格）
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise ValueError(f"无法读取图像: {img}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]
        print(f"图像尺寸: {img_w}x{img_h}")
        
        # P-Net
        boxes = self.detect_pnet(img_rgb, threshold=pnet_threshold)
        print(f"P-Net检测结果: {len(boxes)} 个候选框")
        if len(boxes) > 0:
            scores = [b[4] for b in boxes]
            print(f"  P-Net置信度范围: {min(scores):.3f} - {max(scores):.3f}")
            # 打印前5个候选框的详细信息
            for idx, box in enumerate(boxes[:5]):
                print(f"  候选框{idx+1}: ({box[0]}, {box[1]}) -> ({box[2]}, {box[3]}), 置信度: {box[4]:.3f}, 尺寸: {box[2]-box[0]}x{box[3]-box[1]}")
        else:
            print("  ⚠️  P-Net未检测到任何候选框，可能阈值过高或模型未训练好")
        
        # R-Net
        if self.rnet is not None and len(boxes) > 0:
            boxes = self.detect_rnet(img_rgb, boxes, threshold=rnet_threshold)
            print(f"R-Net检测结果: {len(boxes)} 个候选框")
        
        # O-Net
        if self.onet is not None and len(boxes) > 0:
            boxes, landmarks = self.detect_onet(img_rgb, boxes, threshold=onet_threshold)
            print(f"O-Net检测结果: {len(boxes)} 个人脸")
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
    parser.add_argument('--pnet_threshold', type=float, default=0.7,
                       help='P-Net置信度阈值（默认0.7，越高越严格，减少误检）')
    parser.add_argument('--rnet_threshold', type=float, default=0.8,
                       help='R-Net置信度阈值（默认0.8）')
    parser.add_argument('--onet_threshold', type=float, default=0.9,
                       help='O-Net置信度阈值（默认0.9）')
    parser.add_argument('--nms_threshold', type=float, default=0.4,
                       help='NMS IoU阈值（默认0.4，越低越严格，减少重复检测）')
    
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
    print(f"检测阈值设置: P-Net={args.pnet_threshold}, R-Net={args.rnet_threshold}, O-Net={args.onet_threshold}, NMS={args.nms_threshold}")
    start_time = time.time()
    boxes, landmarks = detector.detect(
        img, 
        pnet_threshold=args.pnet_threshold,
        rnet_threshold=args.rnet_threshold,
        onet_threshold=args.onet_threshold,
        nms_threshold=args.nms_threshold
    )
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
#python detect_mtcnn.py --image ./pengcheng_5.jpg --pnet checkpoints/pnet_best.pth --rnet checkpoints/rnet_best.pth --onet checkpoints/onet_best.pth --output result.jpg --device cuda --pnet_threshold 0.7 --rnet_threshold 0.8 --onet_threshold 0.9 --nms_threshold 0.3 
