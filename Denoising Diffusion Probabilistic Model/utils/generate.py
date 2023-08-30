import torch
import torch.nn as nn
import numpy as np
import tqdm
from utils.loader import output_images

def remove_noise(coefs_0,coefs_1,model,x, t, y):
    coef_0 = coefs_0[t]
    coef_1 = coefs_1[t]
    t_tensor = t * torch.ones((x.shape[0],),dtype=torch.float32).cuda()
    noise = model(x,t_tensor,y)
    print(t)
    print("mean"+str(torch.mean(noise)))
    print("var" +str(torch.var(noise)))
    return coef_1 * (x - coef_0 * noise)

@torch.no_grad()
def generate_image(num_of_images,image_size,coefs_0,coefs_1,sigma,model,y,no_random_range = 0):
    noise = torch.randn((num_of_images,3,image_size,image_size)).cuda()
    num_timestamp = coefs_0.shape[0]
    images_tensor = noise
    for t in reversed(range(num_timestamp)):
        images_tensor = remove_noise(coefs_0,coefs_1,model,images_tensor,t,y)
        if t > no_random_range:
            images_tensor += sigma[t] * torch.randn_like(images_tensor).cuda()
        if t%100==0:
            tensor = images_tensor.permute(0,2,3,1)
            tensor = tensor.cpu().detach()
            tensor = np.array(tensor)
            images = []
            for i in range(tensor.shape[0]):
                image = (tensor[i] + 1)/2
                image = image * 255
                image = np.clip(image,0,255)
                image = image.astype(np.uint8)
                images.append(image)
            output_images(str(t)+"_",images)

def generate_params(betas,alphas,cumprod):
    alphas_shift = torch.tensor([1],dtype = torch.float32).cuda()
    alphas_shift = torch.concat([alphas_shift,cumprod])
    alphas_shift = alphas_shift[0:alphas.shape[0]]
    coefs_0 = betas / ((1 - cumprod)**0.5)
    coefs_1 = (1 / alphas) ** 0.5
    sigma = ((1-alphas_shift)/(1- cumprod)* betas) ** 0.5
    return coefs_0,coefs_1,sigma