import torch
import torch.nn as nn
import numpy as np
from networks.u_net import UNet
from utils.schedule import generate_linear_schedule,generate_cosine_schedule,generate_respacing_schedule
from utils.diffusion import generate_images,generate_images_ddim
import cv2 as cv
from torch.optim import Adam
import tqdm
import os

IMG_SIZE = 96 #图片大小
BATCH_SIZE = 64  #批次大小
NUM_TIMESTAMP = 1000 #Timestamp的个数，表示从原图加多少次噪声变成一张纯噪声图像
MIN_BETA = 0.001
MAX_BETA = 0.02

schedule = generate_linear_schedule(NUM_TIMESTAMP,MIN_BETA,MAX_BETA)
time_map = []
for i in range(NUM_TIMESTAMP):
    time_map.append(i<2)
respacing_schedule = generate_respacing_schedule(schedule,time_map)
# schedule = generate_cosine_schedule(NUM_TIMESTAMP,0.01,0.3) #生成所有的Noise Schedule，这个地方最好进去看看细节
model = UNet(64,1) #建立模型

if os.path.exists("legacy_weights_var/model.pth"):
    model.load_state_dict(torch.load("legacy_weights_var/model.pth"))

model = model.cuda()
model.eval()
y = torch.zeros((BATCH_SIZE,),dtype=torch.long).cuda()
#images = generate_images(model,BATCH_SIZE,IMG_SIZE,NUM_TIMESTAMP,y,schedule,True)
images = generate_images_ddim(model,BATCH_SIZE,IMG_SIZE,time_map,y,0,respacing_schedule)
#images = generate_images(model,BATCH_SIZE,IMG_SIZE,NUM_TIMESTAMP,y,schedule,True)
for i,image in enumerate(images):
    cv.imwrite("results/"+str(i)+".jpg",image)



