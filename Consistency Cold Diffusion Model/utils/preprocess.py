import cv2 as cv
import numpy as np
import torch
import random

def preprocessing(images):
    out_images= []
    for i in range(images.shape[0]):
        image = images[i]
        hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
        h = hsv[:,:,:1]
        sv = hsv[:,:,1:]
        h = h.astype(np.float32)
        h = h + random.normalvariate(0,128)+1800
        h = h % 180
        h = h.clip(0,179)
        h = h.astype(np.uint8)
        hsv = np.concatenate([h,sv],-1)
        out_image = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
        out_image = torch.tensor(out_image,dtype = torch.float32).cuda()
        out_image = out_image / 127.5 - 1
        out_image = out_image.permute(2,0,1)
        out_images.append(out_image)
    out_images = torch.concat(out_images).view(images.shape[0],3,images.shape[1],images.shape[2])
    return out_images
    
        
