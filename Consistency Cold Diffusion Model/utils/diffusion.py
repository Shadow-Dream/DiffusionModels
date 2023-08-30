import torch
import torch.nn as nn
import numpy as np
import tqdm
import cv2 as cv

@torch.no_grad()
def get_kernel(size, std):
    kernel = cv.getGaussianKernel(size,std)
    kernel = torch.tensor(kernel,dtype=torch.float32).cuda()
    kernel = torch.matmul(kernel,kernel.view(1,size))
    kernel = kernel.view(1,1,size,size).repeat(3, 1, 1, 1)
    conv = nn.Conv2d(3, 3, size, padding=(size-1)//2, padding_mode='replicate',bias=False, groups=3)
    conv.weight = nn.Parameter(kernel)
    return conv

def get_kernels(num_timesteps,size, std):
    kernels = []
    for i in range(num_timesteps):
        kernels.append(get_kernel(size, std*(i+1)))
    return kernels

@torch.no_grad()
def get_images_at_timestamp(images,t,kernels):
    batch_size = t.shape[0]
    image_size = images.shape[3]
    out_images = []
    for i in range(batch_size):
        out_images.append(0)
    max_t = int(torch.max(t))
    for i in range(max_t + 1):
        for j in range(batch_size):
            if t[j] == i:
                out_images[j] = images[j]
        images = kernels[i](images)
    images = torch.concat(out_images)
    images = images.view(batch_size,3,image_size,image_size)
    return images
    

