import torch
import torch.nn as nn
import numpy as np
from networks.u_net import UNet
from utils.loader import load_dataset
from utils.diffusion import get_images_at_timestamp,get_kernels
from torch.optim import Adam
from utils.preprocess import preprocessing
import os
import cv2 as cv

IMG_SIZE = 96 #图片大小
BATCH_SIZE = 4 #批次大小
NUM_TIMESTAMP = 1000 #Timestamp的个数，表示从原图加多少次噪声变成一张纯噪声图像

images = load_dataset(IMG_SIZE)[0:BATCH_SIZE] #加载数据集
iters = images.shape[0]//BATCH_SIZE #一次Epoch需要的迭代次数，就是多少个Batch能训练完整个Dataset
model = UNet(64,1) #建立模型
kernels = get_kernels(NUM_TIMESTAMP,3,1)

if os.path.exists("weights/model.pth"):
    model.load_state_dict(torch.load("weights/model.pth"))

model = model.cuda()
model = model.eval() #设置模型为训练模式

#将数据全部扔进显卡
images = preprocessing(images)
t = ((NUM_TIMESTAMP-1)*torch.ones((BATCH_SIZE,))).cuda().to(torch.long)
ori_images = images = get_images_at_timestamp(images,t,kernels)
batch_class = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()
y = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()

indices = []
for i in reversed(range(NUM_TIMESTAMP)):
    if i% 10 == 0:
        indices.append(i)


with torch.no_grad():
    for i in indices:
        t = (i*torch.ones((BATCH_SIZE,))).cuda().to(torch.long)
        predict_images_0 = model(images,t,y)
        if i==0:
            images = images
        else:
            randmap = torch.randn_like(images).cuda()
            predict_images_0 = predict_images_0 + randmap*i/1000
            predict_images_tm0 = get_images_at_timestamp(predict_images_0,t,kernels)
            predict_images_tm1 = get_images_at_timestamp(predict_images_0,t-10,kernels)
            
            images = images - predict_images_tm0 + predict_images_tm1
        images = torch.clamp(images,-1,1)
        print(str(i)+":")
images = images.cpu().detach()
images = images.permute(0,2,3,1)
images = np.array(images)
images = (images+1)*127.5
images = images.astype(np.uint8)
images = np.clip(images,0,255)
for i in range(images.shape[0]):
    image = images[i]
    cv.imwrite("results/"+str(i)+".jpg",image)

ori_images = ori_images.cpu().detach()
ori_images = ori_images.permute(0,2,3,1)
ori_images = np.array(ori_images)
ori_images = (ori_images+1)*127.5
ori_images = ori_images.astype(np.uint8)
ori_images = np.clip(ori_images,0,255)
for i in range(ori_images.shape[0]):
    ori_image = ori_images[i]
    cv.imwrite("results/ori_"+str(i)+".jpg",ori_image)













