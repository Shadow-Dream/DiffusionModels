import torch
import torch.nn as nn
import numpy as np
from networks.u_net import UNet
import os
from utils.diffusion import generate_images
import cv2 as cv

DATASET_END = 32768
DATASET_BEGIN = 0
IMG_SIZE = 96 #图片大小
BATCH_SIZE = 64 #批次大小


NUM_SCALES = 2
SIGMA_MAX = 160
SIGMA_MIN = 0.002
RHO = 7

model = UNet(96,1) #建立模型
model = model.cuda()
model = model.eval()

if os.path.exists("weights/target_model.pth"):
    model.load_state_dict(torch.load("weights/target_model.pth"))

y = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()
images=  generate_images(model,BATCH_SIZE,IMG_SIZE,y,False,NUM_SCALES,SIGMA_MIN,SIGMA_MAX,RHO)
for i in range(images.shape[0]):
    image= images[i]
    cv.imwrite("results/"+str(i)+".jpg",image)








