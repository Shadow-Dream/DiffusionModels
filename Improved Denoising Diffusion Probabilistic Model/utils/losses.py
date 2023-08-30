import torch
import torch.nn as nn
import math

LOG2 = math.log(2)
def kl_divergence(mean1, logvar1, mean2, logvar2):
    batch_size = mean1.shape[0]
    logvar1 = logvar1.view(batch_size,1,1,1)
    logvar2 = logvar2.view(batch_size,1,1,1)
    div = 0.5 * (logvar2 - logvar1+ torch.exp(logvar1 - logvar2)+ ((mean1 - mean2) ** 2) * torch.exp(-logvar2) -1)
    div = torch.mean(torch.mean(torch.mean(div,-1),-1),-1) / LOG2
    return div

def standard_normal_cdf(x):
    return 0.5 * (1 + torch.tanh(((2.0 / torch.pi)**0.5) * (x + 0.044715 * torch.pow(x, 3))))

def log_likelihood(mean_0, mean_t, logvar):
    batch_size = mean_0.shape[0]
    logvar = logvar / 2
    logvar = logvar.view(batch_size,1,1,1)
    centered_x = mean_0 - mean_t
    inv_stdv = torch.exp(-logvar)
    plus_in = inv_stdv * (centered_x + 1 / 255)
    cdf_plus = standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1 / 255)
    cdf_min = standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        mean_0 < -0.999,
        log_cdf_plus,
        torch.where(mean_0 > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    log_probs = torch.mean(torch.mean(torch.mean(log_probs,-1),-1),-1)
    log_probs = log_probs/LOG2
    return log_probs
