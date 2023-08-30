import torch
import torch.nn as nn
import numpy as np
from networks.u_net import UNet
from utils.generate import generate_image,generate_params
from utils.schedule import generate_schedule
from utils.loader import load_dataset

import cv2 as cv

import tqdm
import os

device = torch.device("cuda")
model = UNet(96,1) #建立模型
model.load_state_dict(torch.load("weights/model.pth"))
model = model.to(device)
IMG_SIZE = 96 #图片大小
BATCH_SIZE = 64  #批次大小
BETA_MAX = 0.02 #这里跟原论文可能存在一些差异，我所使用的Noise Schedule是随着t变化的，并不一直都是0.9999，这里的Beta就是1-Alpha
BETA_MIN = 0.0001 #最小的Noise Schedule，中间从0.02到0.0001采用简单的线性过渡
NUM_TIMESTAMP = 1000 #Timestamp的个数，表示从原图加多少次噪声变成一张纯噪声图像

betas,alphas,cumprod = generate_schedule(BETA_MIN,BETA_MAX,NUM_TIMESTAMP)
coefs_0,coefs_1,sigma = generate_params(betas,alphas,cumprod)

@torch.no_grad()
def test_function():
    betas,alphas,cumprod = generate_schedule(BETA_MIN,BETA_MAX,NUM_TIMESTAMP)
    coefs_0,coefs_1,sigma = generate_params(betas,alphas,cumprod)
    image = load_dataset(IMG_SIZE,1,2)
    image = image[0]
    image = torch.tensor(image).cuda()
    image = image.view(1,3,IMG_SIZE,IMG_SIZE)
    alphas = torch.tensor(alphas).cuda()
    betas = torch.tensor(betas).cuda()
    now_image = image
    for i in range(NUM_TIMESTAMP):
        noise = torch.randn_like(image).cuda()
        t = torch.tensor([i],dtype = torch.float32).cuda()
        y = torch.tensor([0],dtype = torch.int).cuda()
        batch_alphas_cumprod = cumprod[i]
        batch_sqrt = (batch_alphas_cumprod**0.5).view(BATCH_SIZE,1,1,1)
        batch_one_minus_sqrt = ((1-batch_alphas_cumprod)**0.5).view(BATCH_SIZE,1,1,1)
        batch_images_add_noise = torch.mul(batch_sqrt, image) + torch.mul(batch_one_minus_sqrt,noise)
        predict_noise = model(batch_images_add_noise,t,y)
        print("1 mean"+str(torch.mean(predict_noise)))
        print("1 var"+str(torch.var(predict_noise)))
        print("loss"+str(torch.mean(predict_noise-noise)))
        noise = torch.randn_like(image).cuda()
        now_image = alphas[i]**0.5 * now_image + (1-alphas[i])**0.5 * noise
        predict_noise = model(now_image,t,y)
        print("2 mean"+str(torch.mean(predict_noise)))
        print("2 var"+str(torch.var(predict_noise)))


    print("picturemean"+str(torch.mean(now_image)))
    print("picturevar"+str(torch.var(now_image)))

    now_image=torch.randn_like(now_image).cuda()
    for i in reversed(range(NUM_TIMESTAMP)):
        t = torch.tensor([i],dtype = torch.float32).cuda()
        y = torch.tensor([0],dtype = torch.int).cuda()
        noise = model(now_image,t,y)
        print("1 mean"+str(torch.mean(noise)))
        print("1 var"+str(torch.var(noise)))
        coef_0 = coefs_0[i]
        coef_1 = coefs_1[i]
        now_image = coef_1 * (now_image - coef_0 * noise) + sigma[i] * torch.randn_like(now_image).cuda()
        print("2 mean"+str(torch.mean(now_image)))
        print("2 var"+str(torch.var(now_image)))
    now_image = (now_image + 1)*127.5
    now_image = now_image.cpu().detach()
    now_image = np.array(now_image)
    now_image = np.clip(now_image,0,255)
    now_image = now_image.astype(np.uint8)
    cv.imwrite("/root/dev/DDPM/test.jpg",now_image)

#test_function()
generate_image(BATCH_SIZE,IMG_SIZE,coefs_0,coefs_1,sigma,model,torch.zeros((BATCH_SIZE,),dtype = torch.int).cuda())