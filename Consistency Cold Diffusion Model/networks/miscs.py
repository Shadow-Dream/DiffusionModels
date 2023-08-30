import torch
import torch.nn as nn

class Downsample(nn.Module):#使用卷积进行降采样（图片大小缩小一倍）
    def __init__(self, in_channels):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
    
    def forward(self, x):
        return self.conv_0(x)

class Upsample(nn.Module):#使用上采样+卷积进行上采样，不使用反卷积是因为反卷积会导致棋盘效应，细节参见Distill棋盘效应论文
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_0 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv_0(x)
        return x

def silu(x):#silu激活函数
    return torch.sigmoid(x) * x