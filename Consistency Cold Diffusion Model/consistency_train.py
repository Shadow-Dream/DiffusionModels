import torch
from networks.u_net import UNet
from utils.loader import load_dataset
from utils.diffusion import get_images_at_timestamp,get_kernels
from utils.preprocess import preprocessing
from torch.optim import Adam
import tqdm
import os

IMG_SIZE = 96
BATCH_SIZE = 4
EPOCHES = 100
NUM_TIMESTAMP = 1000
LEARNING_RATE = 0.0001

def get_coef(t,e0 = 10,e1 = 2,e2 = 3):
    t = ((t + 1)/NUM_TIMESTAMP).view(t.shape[0],1,1,1)

    c0 = (1 - t)**e0
    c1 = (1+(1-t)**e1)/(1+(1-t)**e2) - c0

    return c0,c1

def remove_noise(model,x,t,y):
    c0,c1 = get_coef(t)
    return c0 * x + c1 * model(x,t,y)

def get_ema(src,dst,epoch):
    ratio = epoch/EPOCHES
    ratio = ((1 - ratio)**2)*0.9+0.1
    return src * ratio + dst * (1-ratio)

images = load_dataset(IMG_SIZE)
iters = images.shape[0]//BATCH_SIZE
kernels = get_kernels(NUM_TIMESTAMP,3,1)

model = UNet(64,1)
ema_model = UNet(64,1)
model = model.cuda()
ema_model = ema_model.cuda()

if os.path.exists("weights/model.pth"):
    model.load_state_dict(torch.load("weights/model.pth"))
if os.path.exists("weights/ema_model.pth"):
    ema_model.load_state_dict(torch.load("weights/ema_model.pth"))
else:
    with torch.no_grad():
        for src,dst in zip(model.parameters(),ema_model.parameters()):
            dst.copy_(src)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
model = model.train()

batch_class = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()
y = torch.zeros((BATCH_SIZE,),dtype=torch.int).cuda()
total_loss = 0
for epoch in range(EPOCHES):
    #tqdm是进度条
    process = tqdm.tqdm(range(iters))
    for iter in process:
        #获取当前BATCH的图像数据
        batch_offset = iter * BATCH_SIZE
        batch_images_0 = preprocessing(images[batch_offset:batch_offset+BATCH_SIZE])
        t = torch.randint(0,NUM_TIMESTAMP,(BATCH_SIZE,)).cuda()
        batch_images_t = get_images_at_timestamp(batch_images_0,t,kernels)
        
        optimizer.zero_grad()

        predict_images_0 = remove_noise(model,batch_images_t,t,y)

        loss = torch.mean(torch.mean(torch.mean((batch_images_0 - predict_images_0)**2,-1),-1),-1)
        loss.backward(torch.ones_like(loss))

        optimizer.step()

        with torch.no_grad():
            for src,dst in zip(model.parameters(),ema_model.parameters()):
                dst.copy_(get_ema(src,dst,epoch))
        
        total_loss += loss
        process.set_postfix(**{
        'loss': float(torch.mean(total_loss/(BATCH_SIZE*epoch + (iter+1)),-1)),
        'epoch':float(epoch)
        })
    #保存权重
    torch.save(model.state_dict(), "weights/model.pth")
    torch.save(ema_model.state_dict(), "weights/ema_model.pth")









