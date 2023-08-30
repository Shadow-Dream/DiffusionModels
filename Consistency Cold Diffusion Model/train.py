import torch
from networks.u_net import UNet
from utils.loader import load_dataset
from utils.diffusion import get_images_at_timestamp,get_kernels
from utils.preprocess import preprocessing
from torch.optim import Adam
import tqdm
import os

IMG_SIZE = 96 #图片大小
BATCH_SIZE = 4 #批次大小
EPOCHES = 100 #训练总批次
NUM_TIMESTAMP = 1000 #Timestamp的个数，表示从原图加多少次噪声变成一张纯噪声图像
LEARNING_RATE = 0.0001 #学习率

images = load_dataset(IMG_SIZE) #加载数据集
iters = images.shape[0]//BATCH_SIZE #一次Epoch需要的迭代次数，就是多少个Batch能训练完整个Dataset
model = UNet(64,1) #建立模型
kernels = get_kernels(NUM_TIMESTAMP,3,1)

if os.path.exists("weights/model.pth"):
    model.load_state_dict(torch.load("weights/model.pth"))

model = model.cuda()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE) #Adam梯度下降器
model = model.train() #设置模型为训练模式

#将数据全部扔进显卡
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
        batch_images_0 = preprocessing(images[batch_offset:batch_offset+BATCH_SIZE])
        t = torch.randint(0,NUM_TIMESTAMP,(BATCH_SIZE,)).cuda()
        batch_images_t = get_images_at_timestamp(batch_images_0,t,kernels)
        
        optimizer.zero_grad()

        predict_images_0 = model(batch_images_t,t,y)

        loss = torch.mean(torch.mean(torch.mean((batch_images_0 - predict_images_0)**2,-1),-1),-1)

        loss.backward(torch.ones_like(loss))
        optimizer.step()

        #计算总损失并显示平均损失
        total_loss+=loss
        process.set_postfix(**{
        'loss': float(torch.mean(total_loss/(iter+1),-1)),
        'epoch':float(epoch)
        })
    #保存权重
    torch.save(model.state_dict(), "weights/model.pth")












