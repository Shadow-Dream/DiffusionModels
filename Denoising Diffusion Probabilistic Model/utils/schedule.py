import numpy as np
import torch

#生成Schedule，按照论文中的描述将一些必要的量预先计算，避免训练的时候再计算造成性能降低
def generate_schedule(min,max,num):
    betas = torch.tensor(torch.linspace(min,max,num,dtype=torch.float32)).cuda()
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas,0)#这个函数是计算乘积，cumprod[i] = alphas[0] * alphas[1] * ... * alphas[i]
    return betas,alphas,alphas_cumprod

def generate_cos_schedule(min,max,num):
    alphas = torch.tensor(torch.linspace(0,1,num,dtype=torch.float32)).cuda()
    return