# encoding:utf-8
"""
MTCNN模型定义
包含P-Net, R-Net, O-Net三个网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PNet(nn.Module):
    """Proposal Network - 快速生成候选框"""
    def __init__(self):
        super(PNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(32)
        
        # 分类分支：判断是否为人脸
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1)
        # 回归分支：边界框回归
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        
        # 分类输出 (batch, 2, h, w)
        cls = self.conv4_1(x)
        cls = F.softmax(cls, dim=1)
        
        # 回归输出 (batch, 4, h, w)
        bbox = self.conv4_2(x)
        
        return cls, bbox


class RNet(nn.Module):
    """Refine Network - 精炼候选框"""
    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2, stride=1)
        self.prelu3 = nn.PReLU(64)
        
        # 计算特征图尺寸: 24 -> conv1(22) -> pool1(10) -> conv2(8) -> pool2(3) -> conv3(2)
        # 最终: 64 * 2 * 2 = 256
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.prelu4 = nn.PReLU(128)
        
        # 分类分支
        self.fc2_1 = nn.Linear(128, 2)
        # 回归分支
        self.fc2_2 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.prelu4(self.fc1(x))
        
        cls = F.softmax(self.fc2_1(x), dim=1)
        bbox = self.fc2_2(x)
        
        return cls, bbox


class ONet(nn.Module):
    """Output Network - 最终输出和关键点"""
    def __init__(self):
        super(ONet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.prelu4 = nn.PReLU(128)
        
        # 计算特征图尺寸: 48 -> conv1(46) -> pool1(22) -> conv2(20) -> pool2(9) -> conv3(7) -> pool3(3) -> conv4(2)
        # 最终: 128 * 2 * 2 = 512
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU(256)
        
        # 分类分支
        self.fc2_1 = nn.Linear(256, 2)
        # 回归分支
        self.fc2_2 = nn.Linear(256, 4)
        # 关键点分支（5个关键点，每个2个坐标）
        self.fc2_3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu2(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu3(self.conv3(x))
        x = self.pool3(x)
        x = self.prelu4(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.prelu5(self.fc1(x))
        
        cls = F.softmax(self.fc2_1(x), dim=1)
        bbox = self.fc2_2(x)
        landmarks = self.fc2_3(x)
        
        return cls, bbox, landmarks


def get_model(model_type='pnet'):
    """获取指定类型的模型"""
    if model_type.lower() == 'pnet':
        return PNet()
    elif model_type.lower() == 'rnet':
        return RNet()
    elif model_type.lower() == 'onet':
        return ONet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试P-Net
    pnet = PNet().to(device)
    x = torch.randn(1, 3, 12, 12).to(device)
    cls, bbox = pnet(x)
    print(f"P-Net输出 - 分类: {cls.shape}, 边界框: {bbox.shape}")
    
    # 测试R-Net
    rnet = RNet().to(device)
    x = torch.randn(1, 3, 24, 24).to(device)
    cls, bbox = rnet(x)
    print(f"R-Net输出 - 分类: {cls.shape}, 边界框: {bbox.shape}")
    
    # 测试O-Net
    onet = ONet().to(device)
    x = torch.randn(1, 3, 48, 48).to(device)
    cls, bbox, landmarks = onet(x)
    print(f"O-Net输出 - 分类: {cls.shape}, 边界框: {bbox.shape}, 关键点: {landmarks.shape}")
