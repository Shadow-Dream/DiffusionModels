import torch
import torch.nn as nn
import numpy as np
from networks.u_net import UNet
from utils.loader import load_dataset
from utils.schedule import generate_linear_schedule,generate_cosine_schedule
from utils.diffusion import get_images_at_timestamp,get_real_images,get_predict_images_noise_log_var
from utils.losses import kl_divergence,log_likelihood
from torch.optim import Adam
import tqdm
import os

IMG_SIZE = 96 #图片大小
BATCH_SIZE = 64 #批次大小
EPOCHES = 12 #训练总批次
NUM_TIMESTAMP = 1000 #Timestamp的个数，表示从原图加多少次噪声变成一张纯噪声图像
LEARNING_RATE = 0.0001 #学习率
MIN_BETA = 0.001
MAX_BETA = 0.02

images = load_dataset(IMG_SIZE) #加载数据集
iters = images.shape[0]//BATCH_SIZE #一次Epoch需要的迭代次数，就是多少个Batch能训练完整个Dataset
schedule = generate_linear_schedule(NUM_TIMESTAMP,MIN_BETA,MAX_BETA) #生成所有的Noise Schedule，这个地方最好进去看看细节
model = UNet(64,1) #建立模型

if os.path.exists("weights/model.pth"):
    model.load_state_dict(torch.load("weights/model.pth"))

model = model.cuda()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE) #Adam梯度下降器
model = model.train() #设置模型为训练模式

#将数据全部扔进显卡
images = torch.tensor(images).cuda()
batch_class = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()
y = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()
for epoch in range(EPOCHES):
    #tqdm是进度条
    process = tqdm.tqdm(range(iters))
    #total_loss只有显示的作用
    total_loss = 0
    for iter in process:
        #获取当前BATCH的图像数据
        batch_offset = iter * BATCH_SIZE
        batch_images_0 = images[batch_offset:batch_offset+BATCH_SIZE]
        t = torch.randint(0,NUM_TIMESTAMP,(BATCH_SIZE,)).cuda()

        batch_images_t,batch_noise = get_images_at_timestamp(batch_images_0,t,schedule)
        real_images = get_real_images(batch_images_0,batch_images_t,t,schedule)
        real_log_var = schedule.real_log_var[t]

        optimizer.zero_grad()
        
        predict_images,predict_noise,predict_log_var = get_predict_images_noise_log_var(model,batch_images_t,t,y,schedule)
        
        kl = kl_divergence(real_images,real_log_var,predict_images,predict_log_var)
        like = - log_likelihood(batch_images_0,predict_images,predict_log_var)
        mse = torch.mean(torch.mean(torch.mean((batch_noise - predict_noise)**2,-1),-1),-1)
        var = torch.var(predict_noise)
        normal_like = torch.mean(predict_noise)**2 + var + 1 / var - 2
        # loss = torch.where((t == 0), like, kl) + mse + normal_like
        loss = torch.where((t == 0), like, kl) + mse + normal_like
        loss.backward(torch.ones_like(loss))
        optimizer.step()

        #计算总损失并显示平均损失
        total_loss+=loss
        process.set_postfix(**{
        'loss': float(torch.mean(total_loss/(iter+1),-1)),
        'mse': float(torch.mean(mse,-1)),
        'kl/like': float(torch.mean(torch.where((t == 0), like, kl),-1)),
        'epoch':float(epoch),
        'mean':float(torch.mean(predict_noise)),
        'var':float(torch.var(predict_noise))
        })
    #保存权重
    torch.save(model.state_dict(), "weights/model.pth")
    if epoch%5 == 0:
        torch.save(model.state_dict(), "weights/model"+ str(epoch)+".pth")












