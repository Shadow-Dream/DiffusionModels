import torch
import torch.nn as nn

#Timestamp Embedding层 embedding后timestamp会由一个数字转化为一个向量，这会使其变为高频特征，神经网络更容易辨识
class TimeStampEmbedding(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        self.channels = channels
        self.scale = scale

    # 就是将t拓展为一个切比雪夫多项式，其中scale的作用是将这个t和其它t区分开来，不然全部挤在一堆也不好分辨
    # 这里有1000个timestamp，取ln(1000)就是可以的，方便起见就随便传了个10进去了
    def forward(self, x):
        embedding_vector = torch.exp(-2 * self.scale * torch.arange(self.channels // 2) / self.channels).cuda()
        embedding_vector = torch.outer(x, embedding_vector)
        embedding_vector = torch.cat((embedding_vector.sin(), embedding_vector.cos()), -1)
        return embedding_vector