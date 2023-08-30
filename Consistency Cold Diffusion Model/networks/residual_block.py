import torch
import torch.nn as nn
from networks.self_attention import SelfAttention
from networks.miscs import silu

#残差网络层，就是几层卷积组成的，但是其中有一个Shortcut过程
#in_channels 输入的通道数
#out_channels 输出的通道数
#time_emb_dim timestamp的embedding层数，详情参见timestamp embedding层定义
#num_classes 类别个数，表示我们要生成哪个类别的图片，比如猫是1，狗是0，这个要在训练的时候就包含，如果不想要类型就全部填一个值就好
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes):
        super().__init__()

        self.norm_11 = nn.GroupNorm(32, in_channels)
        self.conv_11 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm_12 = nn.GroupNorm(32, out_channels)
        self.drop_12 = nn.Dropout(0.1)
        self.conv_12 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.conv_21 = nn.Conv2d(in_channels, out_channels, 1)

        self.time_dense_1  = nn.Linear(time_emb_dim, out_channels)
        self.class_dense_1 = nn.Embedding(num_classes, out_channels)

    def forward(self, x, t, c):
        
        way1 = self.norm_11(x)
        way1 = silu(way1)
        way1 = self.conv_11(way1)

        way2 = self.conv_21(x)

        feature_bias = self.time_dense_1(silu(t))[:, :, None, None] + self.class_dense_1(c)[:, :, None, None]
        way1 = way1 + feature_bias

        way1 = self.norm_12(way1)
        way1 = silu(way1)
        way1 = self.drop_12(way1)
        way1 = self.conv_12(way1)

        return way1 + way2

#自注意力残差网络层，就是常规残差网络最后加上一个自注意力层
class AttentionResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes):
        super().__init__()

        self.norm_11 = nn.GroupNorm(32, in_channels)
        self.conv_11 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

        self.norm_12 = nn.GroupNorm(32, out_channels)
        self.drop_12 = nn.Dropout(0.1)
        self.conv_12 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        self.conv_21 = nn.Conv2d(in_channels, out_channels, 1)

        self.time_dense_1  = nn.Linear(time_emb_dim, out_channels)
        self.class_dense_1 = nn.Embedding(num_classes, out_channels)
        self.atten = SelfAttention(out_channels)

    def forward(self, x, t, c):
        way1 = self.norm_11(x)
        way1 = silu(way1)
        way1 = self.conv_11(way1)

        way2 = self.conv_21(x)

        feature_bias = self.time_dense_1(silu(t))[:, :, None, None] + self.class_dense_1(c)[:, :, None, None]
        way1 = way1 + feature_bias

        way1 = self.norm_12(way1)
        way1 = silu(way1)
        way1 = self.drop_12(way1)
        way1 = self.conv_12(way1)

        attention = self.atten(way1 + way2)
        return attention