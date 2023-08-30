import torch
import torch.nn as nn
import numpy as np
from networks.u_net import UNet
from utils.loader import load_dataset
from utils.schedule import generate_schedule
from torch.optim import Adam
import tqdm
import os
import torchvision

device = torch.device("cuda") #使用GPU进行训练

DATASET_END = 32768
DATASET_BEGIN = 0
IMG_SIZE = 96 #图片大小
BATCH_SIZE = 64 #批次大小
EPOCHES = 300 #训练总批次
BETA_MAX = 0.02 #这里跟原论文可能存在一些差异，我所使用的Noise Schedule是随着t变化的，并不一直都是0.9999，这里的Beta就是1-Alpha
BETA_MIN = 0.0001 #最小的Noise Schedule，中间从0.02到0.0001采用简单的线性过渡
NUM_TIMESTAMP = 1000 #Timestamp的个数，表示从原图加多少次噪声变成一张纯噪声图像
LEARNING_RATE = 0.0001 #学习率

images = load_dataset(IMG_SIZE,DATASET_BEGIN,DATASET_END) #加载数据集
iters = images.shape[0]//BATCH_SIZE #一次Epoch需要的迭代次数，就是多少个Batch能训练完整个Dataset
_,_,alphas_cumprod = generate_schedule(BETA_MIN,BETA_MAX,NUM_TIMESTAMP) #生成所有的Noise Schedule，这个地方最好进去看看细节
model = UNet(96,1) #建立模型

if os.path.exists("weights/model.pth"):
    model.load_state_dict(torch.load("weights/model.pth"))

model = model.to(device)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE) #Adam梯度下降器
model = model.train() #设置模型为训练模式

#将数据全部扔进显卡
images = torch.tensor(images).cuda()
batch_class = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()

for epoch in range(EPOCHES):
    #tqdm是进度条
    process = tqdm.tqdm(range(iters))
    #total_loss只有显示的作用
    total_loss = 0
    for iter in process:
        #获取当前BATCH的图像数据
        batch_offset = iter * BATCH_SIZE
        batch_images = images[batch_offset:batch_offset+BATCH_SIZE]
        #生成高斯噪声
        batch_noise = torch.randn_like(batch_images).cuda()
        #随机取几个timestamp来训练，因为整个1000个步骤全部拿来练需要太长时间，这样做的依据是神经网络具有连续性(我认为是这样)
        t = torch.randint(0,NUM_TIMESTAMP,(BATCH_SIZE,)).cuda()
        
        #取出这几个timestamp对应的schedule
        batch_alphas_cumprod = alphas_cumprod[t]
        batch_sqrt = (batch_alphas_cumprod**0.5).view(BATCH_SIZE,1,1,1)
        batch_one_minus_sqrt = ((1-batch_alphas_cumprod)**0.5).view(BATCH_SIZE,1,1,1)
        #按照论文公式施加噪声
        batch_images_add_noise = torch.mul(batch_sqrt, batch_images) + torch.mul(batch_one_minus_sqrt,batch_noise)

        # 清空优化器的梯度
        optimizer.zero_grad()
        # 前向计算得到预测的梯度
        predict_noise = model(batch_images_add_noise,t,batch_class)
        # 计算原噪声和预测出来的噪声的差异
        loss = torch.mean(nn.functional.mse_loss(predict_noise,batch_noise))
        
        # 反向传播计算梯度
        loss.backward()
        # 应用梯度下降
        optimizer.step()

        #计算总损失并显示平均损失
        total_loss+=loss
        process.set_postfix(**{'loss': float(total_loss/(iter+1)),"mean":float(torch.mean(predict_noise)),"var":float(torch.var(predict_noise)) })
    #保存权重
    torch.save(model.state_dict(), "weights/model.pth")












