import torch
import torch.nn as nn
from networks.residual_block import AttentionResidualBlock,ResidualBlock
from networks.timestamp_embedding import TimeStampEmbedding
from networks.miscs import Upsample,Downsample,silu

#nn.Module表示神经网络的一个层，一个层可以由很多个其他的层组成
#其中在init中，我们可以初始化所有其他层
#forward函数表示前向计算，也就是调用神经网络这一层的过程，里面只能使用nn的东西，用了np啥的就会寄（py的内建类型，比如list啥的可以用）
#tensorflow之类的应该也是这样的

# 这里的UNet结构被我改过，做了一点阉割，每一层的残差块改成了两个，降采样次数改为了三次
# 中间shortcut过去的东西也被我改过，我传了残差卷积之后的结果过去，我感觉这样比较好
class UNet(nn.Module):
    def __init__(
        self, channels=64, num_classes=None
    ):
        super().__init__()
        self.num_classes = num_classes
        
        self.time_embed = TimeStampEmbedding(channels, 10)
        self.time_dense_1 = nn.Linear(channels, channels*4)
        self.time_dense_2 = nn.Linear(channels*4, channels*4)
    
        self.conv_0 = nn.Conv2d(3, channels, 3, 1, 1)
        
        self.down_conv_00 = ResidualBlock(channels, channels,channels*4, num_classes)#64
        self.down_conv_01 = ResidualBlock(channels, channels,channels*4, num_classes)
        self.down_down_0 = Downsample(channels)

        self.down_conv_10 = AttentionResidualBlock(channels, channels,channels*4, num_classes)#32
        self.down_conv_11 = AttentionResidualBlock(channels, channels*2,channels*4, num_classes)
        self.down_down_1 = Downsample(channels*2)

        self.down_conv_20 = ResidualBlock(channels*2, channels*2,channels*4, num_classes)#16
        self.down_conv_21 = ResidualBlock(channels*2, channels*4,channels*4, num_classes)
        self.down_down_2 = Downsample(channels*4)

        self.mid_conv_0 = AttentionResidualBlock(channels*4, channels*4,channels*4, num_classes)#8
        self.mid_conv_1 = ResidualBlock(channels*4, channels*4,channels*4, num_classes)

        self.up_up_0 = Upsample(channels*4)
        self.up_conv_00 = ResidualBlock(channels*8, channels*4,channels*4, num_classes)#16
        self.up_conv_01 = ResidualBlock(channels*4, channels*2,channels*4, num_classes)

        self.up_up_1 = Upsample(channels*2)
        self.up_conv_10 = AttentionResidualBlock(channels*4, channels*2,channels*4, num_classes)#32
        self.up_conv_11 = AttentionResidualBlock(channels*2, channels,channels*4, num_classes)

        self.up_up_2 = Upsample(channels)
        self.up_conv_20 = ResidualBlock(channels*2, channels,channels*4, num_classes)#64
        self.up_conv_21 = ResidualBlock(channels, channels,channels*4, num_classes)
        
        self.norm_0 = nn.GroupNorm(32, channels)
        self.conv_1 = nn.Conv2d(channels, 3, 3, 1, 1)
    
    def forward(self, x, t, y):
        t = self.time_embed(t)
        t = self.time_dense_1(t)
        t = silu(t)
        t = self.time_dense_2(t)
        shortcut = []

        down0 = self.conv_0(x)

        down0 = self.down_conv_00(down0, t, y)
        down0 = self.down_conv_01(down0, t, y)
        shortcut.append(down0)
        down0 = self.down_down_0(down0)

        down1 = self.down_conv_10(down0, t, y)
        down1 = self.down_conv_11(down1, t, y)
        shortcut.append(down1)
        down1 = self.down_down_1(down1)

        down2 = self.down_conv_20(down1, t, y)
        down2 = self.down_conv_21(down2, t, y)
        shortcut.append(down2)
        down2 = self.down_down_2(down2)

        mid0 = self.mid_conv_0(down2,t,y)
        mid1 = self.mid_conv_1(mid0,t,y)

        up0 = self.up_up_0(mid1)
        up0 = torch.cat([up0, shortcut.pop()], 1)
        up0 = self.up_conv_00(up0,t,y)
        up0 = self.up_conv_01(up0,t,y)

        up1 = self.up_up_1(up0)
        up1 = torch.cat([up1, shortcut.pop()], 1)
        up1 = self.up_conv_10(up1,t,y)
        up1 = self.up_conv_11(up1,t,y)

        up2 = self.up_up_2(up1)
        up2 = torch.cat([up2, shortcut.pop()], 1)
        up2 = self.up_conv_20(up2,t,y)
        up2 = self.up_conv_21(up2,t,y)

        norm = self.norm_0(up2)
        image_out = self.conv_1(norm)
        return image_out