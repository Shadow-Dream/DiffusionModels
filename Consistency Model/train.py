import torch
import torch.nn as nn
import numpy as np
from networks.u_net import UNet
from utils.loader import load_dataset
from torch.optim import Adam
import tqdm
import os
from utils.diffusion import consistency_losses,ema_avg_function

DATASET_END = 32768
DATASET_BEGIN = 0
IMG_SIZE = 96 #图片大小
BATCH_SIZE = 64 #批次大小
EPOCHES = 300 #训练总批次
LEARNING_RATE = 0.00001 #学习率


NUM_SCALES = 160
SIGMA_MAX = 160
SIGMA_MIN = 0.002
RHO = 7

images = load_dataset(IMG_SIZE,DATASET_BEGIN,DATASET_END) #加载数据集
iters = images.shape[0]//BATCH_SIZE #一次Epoch需要的迭代次数，就是多少个Batch能训练完整个Dataset

model = UNet(96,1) #建立模型
target_model = UNet(96,1) #建立模型
model = model.cuda()
target_model = target_model.cuda()
model = model.train()
target_model = target_model.train()

if os.path.exists("weights/model.pth"):
    model.load_state_dict(torch.load("weights/model.pth"))

for dst, src in zip(target_model.parameters(), model.parameters()):
    dst.data.copy_(src.data)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE) #Adam梯度下降器

images = torch.tensor(images).cuda()
y = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()

for epoch in range(EPOCHES):
    process = tqdm.tqdm(range(iters))
    total_loss = 0
    for iter in process:
        batch_offset = iter * BATCH_SIZE
        if batch_offset+BATCH_SIZE > DATASET_END - DATASET_BEGIN:
            break
        batch_images = images[batch_offset:batch_offset+BATCH_SIZE]

        optimizer.zero_grad()

        loss = consistency_losses(model,batch_images,y,target_model,NUM_SCALES,SIGMA_MAX,SIGMA_MIN,RHO)

        loss.backward(torch.ones_like(loss))
        optimizer.step()

        with torch.no_grad():
            for dst, src in zip(target_model.parameters(), model.parameters()):
                dst.data.copy_(ema_avg_function(src.data,dst.data,epoch,EPOCHES))

        total_loss+=loss
        process.set_postfix(**{'loss': float(torch.mean(total_loss/(iter+1))) })

    torch.save(model.state_dict(), "weights/model.pth")
    torch.save(target_model.state_dict(), "weights/target_model.pth")












