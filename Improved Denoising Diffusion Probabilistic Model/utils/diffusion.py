import torch
import torch.nn as nn
import numpy as np
import tqdm
import cv2 as cv
from utils.schedule import get_respacing_time_sequence

def get_images_at_timestamp(images_0,t,schedule):
    sqrt_cumprod = schedule.sqrt_cumprod[t]
    sqrt_one_minus_cumprod = schedule.sqrt_one_minus_cumprod[t]
    batch_size = images_0.shape[0]
    batch_noise = torch.randn_like(images_0).cuda()
    batch_images_t = torch.mul(sqrt_cumprod.view(batch_size,1,1,1), images_0) + torch.mul(sqrt_one_minus_cumprod.view(batch_size,1,1,1),batch_noise)
    return batch_images_t,batch_noise

def get_real_images(images_0, images_t, t, schedule):
    real_image_0_coef = schedule.real_image_0_coef[t]
    real_image_t_coef = schedule.real_image_t_coef[t]
    batch_size = images_0.shape[0]
    real_mean = torch.mul(real_image_0_coef.view(batch_size,1,1,1), images_0) + torch.mul(real_image_t_coef.view(batch_size,1,1,1),images_t)
    return real_mean

def get_predict_images_noise_log_var(model, image_t, t, y, schedule,clip = False):
    real_log_var = schedule.real_log_var[t]
    log_betas = schedule.log_betas[t]
    batch_size = image_t.shape[0]
    sqrt_recip_cumprod = schedule.sqrt_recip_cumprod[t].view(batch_size,1,1,1)
    sqrt_recip_cumprod_minus_one = schedule.sqrt_recip_cumprod_minus_one[t].view(batch_size,1,1,1)
    predict_noise,v = model(image_t, t, y)
    v = v.view(v.shape[0])
    predict_log_var = v * log_betas + (1 - v) * real_log_var
    predict_images_0 = sqrt_recip_cumprod*image_t - sqrt_recip_cumprod_minus_one*predict_noise
    if clip==True:
        predict_images_0 = torch.clamp(predict_images_0,-1,1)
    predict_images = get_real_images(predict_images_0,image_t,t,schedule)
    return predict_images,predict_noise,predict_log_var

@torch.no_grad()
def get_images_remove_noise(model, image_t, t, y,schedule,clip = False):
    predict_images,_,predict_log_var = get_predict_images_noise_log_var(model,image_t,t,y,schedule,clip)
    noise = torch.randn_like(image_t).cuda()
    predict_images = predict_images + torch.exp(0.5 * predict_log_var).view(predict_log_var.shape[0],1,1,1) * noise
    return predict_images

def generate_images(model,batch_size,image_size,num_timesteps,y,schedule,clip = False):
    images = torch.randn((batch_size,3,image_size,image_size)).cuda()
    for i in tqdm.tqdm(reversed(range(num_timesteps))):
        t = (i * torch.ones((batch_size,)).cuda()).type(torch.long)
        images = get_images_remove_noise(model,images,t,y,schedule,clip)
    images = images.permute(0,2,3,1)
    images = images.cpu().detach()
    images = np.array(images)
    images = (images+1)*127.5
    images = np.clip(images,0,255)
    images = images.astype(np.uint8)
    images_output = []
    for i in range(images.shape[0]):
        images_output.append(images[i])
    return images_output

@torch.no_grad()
def get_images_remove_noise_ddim(model,image,ori_t,new_t,y,param,schedule):
    batch_size = image.shape[0]
    cumprod = schedule.cumprod[new_t].view(batch_size,1,1,1)
    cumprod_minus_one=schedule.cumprod_minus_one[new_t].view(batch_size,1,1,1)
    sqrt_recip_cumprod = schedule.sqrt_recip_cumprod[new_t].view(batch_size,1,1,1)
    sqrt_recip_cumprod_minus_one = schedule.sqrt_recip_cumprod_minus_one[new_t].view(batch_size,1,1,1)

    predict_noise,_ = model(image, ori_t, y)
    predict_images_0 = sqrt_recip_cumprod*image - sqrt_recip_cumprod_minus_one*predict_noise
    predict_var = (sqrt_recip_cumprod * image - predict_images_0)/sqrt_recip_cumprod_minus_one
    sigma = param * (((1 - cumprod_minus_one) / (1 - cumprod))**0.5)* ((1 - cumprod / cumprod_minus_one)**0.5)

    noise = torch.randn_like(image)
    mean_pred = predict_images_0 * ((cumprod_minus_one)**0.5) + ((1 - cumprod_minus_one - sigma ** 2)**0.5) * predict_var
    sample = mean_pred + sigma * noise
    return sample

def generate_images_ddim(model,batch_size,image_size,time_map,y,param,schedule):
    images = torch.randn((batch_size,3,image_size,image_size),dtype=torch.float32).cuda()
    time_sequence = get_respacing_time_sequence(time_map)
    for new_i,ori_i in reversed(list(enumerate(time_sequence))):
        new_t = (new_i * torch.ones((batch_size,)).cuda()).type(torch.long)
        ori_t = (ori_i * torch.ones((batch_size,)).cuda()).type(torch.long)
        images = get_images_remove_noise_ddim(model,images,ori_t,new_t,y,param,schedule)
        print(torch.mean(images))
    images = images.permute(0,2,3,1)
    images = images.cpu().detach()
    images = np.array(images)
    images = (images+1)*127.5
    images = np.clip(images,0,255)
    images = images.astype(np.uint8)
    images_output = []
    for i in range(images.shape[0]):
        images_output.append(images[i])
    return images_output
