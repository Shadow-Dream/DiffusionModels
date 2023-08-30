import numpy as np
import torch
import torch.nn as nn

def get_scalings(sigma):
    c_skip = 0.25 / (sigma**2 + 0.25)
    c_out = sigma * 0.5 / (sigma**2 + 0.25) ** 0.5
    c_in = 1 / (sigma**2 + 0.25) ** 0.5
    return c_skip, c_out, c_in

def remove_noise(model,images_t,sigmas,y):
    batch_size = images_t.shape[0]
    c_skip, c_out, c_in = get_scalings(sigmas)
    c_skip = c_skip.view(batch_size,1,1,1)
    c_out = c_out.view(batch_size,1,1,1)
    c_in = c_in.view(batch_size,1,1,1)
    rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
    model_output = model(c_in * images_t, rescaled_t, y)
    denoised = c_out * model_output + c_skip * images_t
    return denoised

@torch.no_grad()
def euler(images_t, sigma, sigmam1, images_0):
    batch_size = images_t.shape[0]
    x = images_t
    d = (x - images_0) / sigma.view(batch_size,1,1,1)
    images_t = x + d * (sigmam1 - sigma).view(batch_size,1,1,1)
    return images_t

def consistency_losses(model,images_0,y,target_model,num_scales = 40,sigma_max = 80,sigma_min = 0.002,rho = 7):
        noise = torch.randn_like(images_0)
        
        batch_size = images_0.shape[0]

        indices = torch.randint(0, num_scales - 1, (images_0.shape[0],), device=images_0.device)

        sigma = (sigma_max ** (1 / rho) + indices / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)))**rho

        sigmam1 = (sigma_max ** (1 / rho) + (indices + 1) / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho)))**rho

        images_t = images_0 + noise * sigma.view(batch_size,1,1,1)#直接加噪得到t的图像

        rand_state = torch.get_rng_state()
        predict_images_0 = remove_noise(model, images_t, sigma, y)
        images_tm1 = euler(images_t, sigma, sigmam1, images_0).detach()#欧拉法估计t+1的图像

        torch.set_rng_state(rand_state)
        with torch.no_grad():
            target_images_0 = remove_noise(target_model, images_tm1, sigmam1, y)
        target_images_0 = target_images_0.detach()

        snrs = sigma**-2
        weights = snrs + 4

        loss = torch.mean(torch.mean(torch.mean((predict_images_0 - target_images_0) ** 2,-1),-1),-1) * weights
        return loss

@torch.no_grad()
def ema_avg_function(src_data,dst_data,epoch,epoches):#src来自model
    ratio = ((1 - epoch/epoches)**2) *0.9 + 0.1
    return ratio * src_data + (1 - ratio) * dst_data

@torch.no_grad()
def get_sigmas_karras(num_scales = 40,sigma_min = 0.002,sigma_max = 80,rho = 7):
    ramp = torch.linspace(0, 1, num_scales).cuda()
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = torch.concat([sigmas,torch.zeros((1,)).cuda()])
    return sigmas

@torch.no_grad()
def generate_images(model,batch_size,image_size,y,clip = False,num_scales = 40,sigma_min = 0.002,sigma_max = 80,rho = 7):
    sigmas = get_sigmas_karras(num_scales,sigma_min,sigma_max,rho)
    x = torch.randn((batch_size,3,image_size,image_size),dtype = torch.float32).cuda()*sigma_max
    def denoiser(image_t,sigma):
        image_0 = remove_noise(model,image_t,sigma,y)
        if clip:
            image_0 = torch.clamp(image_0,-1,1)
        return image_0
    
    for i in range(num_scales):
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * torch.ones((batch_size,),dtype = torch.float32).cuda())
        d = (x - denoised) / sigma
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
        print(torch.var(x))
    x =  torch.clamp(x,-1,1)
    x = x.permute(0,2,3,1)
    x = x.cpu().detach()
    x = np.array(x)
    x = (x + 1)*127.5
    x = x.astype(np.uint8)
    return x

