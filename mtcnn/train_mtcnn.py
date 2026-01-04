# encoding:utf-8
"""
MTCNN训练脚本
支持P-Net, R-Net, O-Net的训练
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from mtcnn_model import get_model
from mtcnn_dataset import create_dataloader
import argparse


class MTCNNLoss(nn.Module):
    """MTCNN损失函数"""
    def __init__(self, mode='pnet'):
        super(MTCNNLoss, self).__init__()
        self.mode = mode
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.MSELoss()
        if mode == 'onet':
            self.landmark_loss = nn.MSELoss()
    
    def forward(self, cls_pred, cls_target, bbox_pred, bbox_target, 
                landmark_pred=None, landmark_target=None):
        # 分类损失
        cls_loss = self.cls_loss(cls_pred, cls_target)
        
        # 只对正样本计算边界框损失
        pos_mask = (cls_target == 1)
        if pos_mask.sum() > 0:
            bbox_loss = self.bbox_loss(
                bbox_pred[pos_mask], 
                bbox_target[pos_mask]
            )
        else:
            bbox_loss = torch.tensor(0.0, device=cls_pred.device)
        
        total_loss = cls_loss + 0.5 * bbox_loss
        
        # O-Net还需要关键点损失
        if self.mode == 'onet' and landmark_pred is not None:
            if pos_mask.sum() > 0:
                landmark_loss = self.landmark_loss(
                    landmark_pred[pos_mask],
                    landmark_target[pos_mask]
                )
                total_loss += 0.5 * landmark_loss
            else:
                landmark_loss = torch.tensor(0.0, device=cls_pred.device)
            return total_loss, cls_loss, bbox_loss, landmark_loss
        
        return total_loss, cls_loss, bbox_loss


def train_epoch(model, dataloader, criterion, optimizer, device, mode='pnet'):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    total_landmark_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if mode == 'onet':
            images, cls_target, bbox_target, landmark_target = batch
            images = images.to(device)
            cls_target = cls_target.to(device)
            bbox_target = bbox_target.to(device)
            landmark_target = landmark_target.to(device)
            
            cls_pred, bbox_pred, landmark_pred = model(images)
            loss, cls_loss, bbox_loss, landmark_loss = criterion(
                cls_pred, cls_target, bbox_pred, bbox_target,
                landmark_pred, landmark_target
            )
            total_landmark_loss += landmark_loss.item()
        else:
            images, cls_target, bbox_target = batch
            images = images.to(device)
            cls_target = cls_target.to(device)
            bbox_target = bbox_target.to(device)
            
            cls_pred, bbox_pred = model(images)
            loss, cls_loss, bbox_loss = criterion(
                cls_pred, cls_target, bbox_pred, bbox_target
            )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_bbox_loss += bbox_loss.item()
        
        # 计算准确率
        _, predicted = cls_pred.max(1)
        total += cls_target.size(0)
        correct += predicted.eq(cls_target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx+1}/{len(dataloader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(dataloader)
    avg_cls_loss = total_cls_loss / len(dataloader)
    avg_bbox_loss = total_bbox_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    if mode == 'onet':
        avg_landmark_loss = total_landmark_loss / len(dataloader)
        return avg_loss, avg_cls_loss, avg_bbox_loss, avg_landmark_loss, accuracy
    else:
        return avg_loss, avg_cls_loss, avg_bbox_loss, accuracy


def validate(model, dataloader, criterion, device, mode='pnet'):
    """验证"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if mode == 'onet':
                images, cls_target, bbox_target, landmark_target = batch
                images = images.to(device)
                cls_target = cls_target.to(device)
                bbox_target = bbox_target.to(device)
                landmark_target = landmark_target.to(device)
                
                cls_pred, bbox_pred, landmark_pred = model(images)
                loss, _, _, _ = criterion(
                    cls_pred, cls_target, bbox_pred, bbox_target,
                    landmark_pred, landmark_target
                )
            else:
                images, cls_target, bbox_target = batch
                images = images.to(device)
                cls_target = cls_target.to(device)
                bbox_target = bbox_target.to(device)
                
                cls_pred, bbox_pred = model(images)
                loss, _, _ = criterion(
                    cls_pred, cls_target, bbox_pred, bbox_target
                )
            
            total_loss += loss.item()
            _, predicted = cls_pred.max(1)
            total += cls_target.size(0)
            correct += predicted.eq(cls_target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='MTCNN训练')
    parser.add_argument('--mode', type=str, default='pnet', 
                       choices=['pnet', 'rnet', 'onet'],
                       help='训练的网络类型')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 图像尺寸
    img_sizes = {'pnet': 12, 'rnet': 24, 'onet': 48}
    img_size = img_sizes[args.mode]
    
    # 创建模型
    model = get_model(args.mode).to(device)
    print(f"创建 {args.mode.upper()}-Net 模型")
    
    # 损失函数和优化器
    criterion = MTCNNLoss(mode=args.mode)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 恢复训练
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"从epoch {start_epoch}恢复训练")
    
    # 数据加载器
    print("加载数据...")
    train_loader = create_dataloader(
        args.data_dir, 
        img_size=img_size, 
        mode=args.mode,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # TensorBoard
    writer = SummaryWriter(f'./runs/{args.mode}_train')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    print(f"开始训练 {args.mode.upper()}-Net...")
    best_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch [{epoch+1}/{args.epochs}]')
        
        # 训练
        if args.mode == 'onet':
            loss, cls_loss, bbox_loss, landmark_loss, acc = train_epoch(
                model, train_loader, criterion, optimizer, device, args.mode
            )
            print(f'Train - Loss: {loss:.4f}, Cls: {cls_loss:.4f}, '
                  f'Bbox: {bbox_loss:.4f}, Landmark: {landmark_loss:.4f}, '
                  f'Acc: {acc:.2f}%')
            writer.add_scalar('Train/Loss', loss, epoch)
            writer.add_scalar('Train/ClsLoss', cls_loss, epoch)
            writer.add_scalar('Train/BboxLoss', bbox_loss, epoch)
            writer.add_scalar('Train/LandmarkLoss', landmark_loss, epoch)
            writer.add_scalar('Train/Accuracy', acc, epoch)
        else:
            loss, cls_loss, bbox_loss, acc = train_epoch(
                model, train_loader, criterion, optimizer, device, args.mode
            )
            print(f'Train - Loss: {loss:.4f}, Cls: {cls_loss:.4f}, '
                  f'Bbox: {bbox_loss:.4f}, Acc: {acc:.2f}%')
            writer.add_scalar('Train/Loss', loss, epoch)
            writer.add_scalar('Train/ClsLoss', cls_loss, epoch)
            writer.add_scalar('Train/BboxLoss', bbox_loss, epoch)
            writer.add_scalar('Train/Accuracy', acc, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 保存检查点
        if acc > best_acc:
            best_acc = acc
            checkpoint_path = os.path.join(
                args.save_dir, 
                f'{args.mode}_best.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
            }, checkpoint_path)
            print(f'保存最佳模型: {checkpoint_path} (Acc: {acc:.2f}%)')
        
        # 定期保存
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                args.save_dir, 
                f'{args.mode}_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': acc,
            }, checkpoint_path)
    
    writer.close()
    print("\n训练完成！")


if __name__ == '__main__':
    main()
