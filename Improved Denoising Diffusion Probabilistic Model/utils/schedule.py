import numpy as np
import torch
import math

class Schedule:
    def __init__(self,cumprod,cumprod_plus_one,cumprod_minus_one,sqrt_cumprod,sqrt_one_minus_cumprod,
        betas,alphas,sqrt_alphas,real_image_0_coef,real_image_t_coef,real_var,real_log_var,log_betas,
        sqrt_recip_cumprod,sqrt_recip_cumprod_minus_one,coef_0,coef_1):

        self.cumprod=cumprod
        self.cumprod_plus_one=cumprod_plus_one
        self.cumprod_minus_one=cumprod_minus_one
        self.sqrt_cumprod=sqrt_cumprod
        self.sqrt_one_minus_cumprod=sqrt_one_minus_cumprod
        self.betas=betas
        self.alphas=alphas
        self.sqrt_alphas=sqrt_alphas
        self.real_image_0_coef=real_image_0_coef
        self.real_image_t_coef=real_image_t_coef
        self.real_var=real_var
        self.real_log_var=real_log_var
        self.log_betas=log_betas
        self.sqrt_recip_cumprod=sqrt_recip_cumprod
        self.sqrt_recip_cumprod_minus_one=sqrt_recip_cumprod_minus_one
        self.coef_0 = coef_0
        self.coef_1 = coef_1

def generate_linear_schedule(num,min_beta,max_beta):
    betas = torch.linspace(min_beta,max_beta,num,dtype = torch.float32).cuda()
    alphas = 1 - betas
    cumprod = torch.cumprod(alphas,0)
    cumprod_plus_one = torch.concat([cumprod,torch.tensor([0],dtype=torch.float32).cuda()])[1:num+1]
    cumprod_minus_one = torch.concat([torch.tensor([1],dtype=torch.float32).cuda(),cumprod])[0:num]
    sqrt_cumprod = cumprod ** 0.5
    sqrt_one_minus_cumprod = (1 - cumprod) ** 0.5
    
    sqrt_alphas = alphas**0.5
    real_image_0_coef = betas*((cumprod_minus_one)**0.5)/(1-cumprod)
    real_image_t_coef = (1 - cumprod_minus_one)* sqrt_alphas / (1 - cumprod)

    real_var = betas*(1 - cumprod_minus_one)/(1 - cumprod)
    real_log_var = real_var
    real_log_var[0] = real_var[1]
    real_log_var = torch.log(real_log_var)

    log_betas = torch.log(betas)
    sqrt_recip_cumprod = (1/cumprod)**0.5
    sqrt_recip_cumprod_minus_one = (1 / cumprod - 1)**0.5

    coef_0 = betas/sqrt_one_minus_cumprod
    coef_1 = 1/sqrt_alphas
    
    return Schedule(cumprod.to(torch.float32),
    cumprod_plus_one.to(torch.float32),
    cumprod_minus_one.to(torch.float32),
    sqrt_cumprod.to(torch.float32),
    sqrt_one_minus_cumprod.to(torch.float32),
    betas.to(torch.float32),
    alphas.to(torch.float32),
    sqrt_alphas.to(torch.float32),
    real_image_0_coef.to(torch.float32),
    real_image_t_coef.to(torch.float32),
    real_var.to(torch.float32),
    real_log_var.to(torch.float32),
    log_betas.to(torch.float32),
    sqrt_recip_cumprod.to(torch.float32),
    sqrt_recip_cumprod_minus_one.to(torch.float32),
    coef_0.to(torch.float32),
    coef_1.to(torch.float32)
    )
#生成Schedule，按照论文中的描述将一些必要的量预先计算，避免训练的时候再计算造成性能降低
def generate_cosine_schedule(num,param = 0.008,max_beta = 0.999):
    betas = generate_accurate_betas(num,param,max_beta)
    alphas = 1 - betas
    cumprod = torch.cumprod(alphas,0)
    cumprod_plus_one = torch.concat([cumprod,torch.tensor([0],dtype=torch.float32).cuda()])[1:num+1]
    cumprod_minus_one = torch.concat([torch.tensor([1],dtype=torch.float32).cuda(),cumprod])[0:num]
    sqrt_cumprod = cumprod ** 0.5
    sqrt_one_minus_cumprod = (1 - cumprod) ** 0.5
    
    sqrt_alphas = alphas**0.5
    real_image_0_coef = betas*((cumprod_minus_one)**0.5)/(1-cumprod)
    real_image_t_coef = (1 - cumprod_minus_one)* sqrt_alphas / (1 - cumprod)

    real_var = betas*(1 - cumprod_minus_one)/(1 - cumprod)
    real_log_var = real_var
    real_log_var[0] = real_var[1]
    real_log_var = torch.log(real_log_var)

    log_betas = torch.log(betas)
    sqrt_recip_cumprod = (1/cumprod)**0.5
    sqrt_recip_cumprod_minus_one = (1 / cumprod - 1)**0.5

    coef_0 = betas/sqrt_one_minus_cumprod
    coef_1 = 1/sqrt_alphas
    
    return Schedule(cumprod.to(torch.float32),
    cumprod_plus_one.to(torch.float32),
    cumprod_minus_one.to(torch.float32),
    sqrt_cumprod.to(torch.float32),
    sqrt_one_minus_cumprod.to(torch.float32),
    betas.to(torch.float32),
    alphas.to(torch.float32),
    sqrt_alphas.to(torch.float32),
    real_image_0_coef.to(torch.float32),
    real_image_t_coef.to(torch.float32),
    real_var.to(torch.float32),
    real_log_var.to(torch.float32),
    log_betas.to(torch.float32),
    sqrt_recip_cumprod.to(torch.float32),
    sqrt_recip_cumprod_minus_one.to(torch.float32),
    coef_0.to(torch.float32),
    coef_1.to(torch.float32)
    )

def generate_accurate_betas(num,param,max_beta):
    betas = []
    t2 = 0
    alpha_bar2 = math.cos(param / (1+param) * math.pi / 2) ** 2
    for i in range(num):
        t2 = (i + 1) / num
        alpha_bar1 = alpha_bar2
        alpha_bar2 = math.cos((t2 + param) / (1+param) * math.pi / 2) ** 2
        betas.append(min(1 - alpha_bar2 / alpha_bar1, max_beta))
    return torch.tensor(betas,dtype=torch.float64).cuda()

def generate_respacing_schedule(original_schedule,time_map):
    last_cumprod = 1.0
    respacing_betas = []
    for i, cumprod in enumerate(original_schedule.cumprod):
        if time_map[i]:
            respacing_beta = 1 - cumprod / last_cumprod
            last_cumprod = cumprod
            respacing_betas.append(respacing_beta)
    respacing_betas = torch.tensor(respacing_betas,dtype = torch.float32).cuda()
    return generate_schedule_with_betas(respacing_betas)

def get_respacing_time_sequence(time_map):
    time_sequence = []
    for i,time in enumerate(time_map):
        if time:
            time_sequence.append(i)
    return time_sequence

def generate_schedule_with_betas(betas):
    num = betas.shape[0]
    alphas = 1 - betas
    cumprod = torch.cumprod(alphas,0)
    cumprod_plus_one = torch.concat([cumprod,torch.tensor([0],dtype=torch.float32).cuda()])[1:num+1]
    cumprod_minus_one = torch.concat([torch.tensor([1],dtype=torch.float32).cuda(),cumprod])[0:num]
    sqrt_cumprod = cumprod ** 0.5
    sqrt_one_minus_cumprod = (1 - cumprod) ** 0.5
    
    sqrt_alphas = alphas**0.5
    real_image_0_coef = betas*((cumprod_minus_one)**0.5)/(1-cumprod)
    real_image_t_coef = (1 - cumprod_minus_one)* sqrt_alphas / (1 - cumprod)

    real_var = betas*(1 - cumprod_minus_one)/(1 - cumprod)
    real_log_var = real_var
    real_log_var[0] = real_var[1]
    real_log_var = torch.log(real_log_var)

    log_betas = torch.log(betas)
    sqrt_recip_cumprod = (1/cumprod)**0.5
    sqrt_recip_cumprod_minus_one = (1 / cumprod - 1)**0.5

    coef_0 = betas/sqrt_one_minus_cumprod
    coef_1 = 1/sqrt_alphas
    
    return Schedule(cumprod.to(torch.float32),
    cumprod_plus_one.to(torch.float32),
    cumprod_minus_one.to(torch.float32),
    sqrt_cumprod.to(torch.float32),
    sqrt_one_minus_cumprod.to(torch.float32),
    betas.to(torch.float32),
    alphas.to(torch.float32),
    sqrt_alphas.to(torch.float32),
    real_image_0_coef.to(torch.float32),
    real_image_t_coef.to(torch.float32),
    real_var.to(torch.float32),
    real_log_var.to(torch.float32),
    log_betas.to(torch.float32),
    sqrt_recip_cumprod.to(torch.float32),
    sqrt_recip_cumprod_minus_one.to(torch.float32),
    coef_0.to(torch.float32),
    coef_1.to(torch.float32)
    )