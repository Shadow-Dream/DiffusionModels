import os
import cv2 as cv
import numpy as np

#从数据集中加载所有图片
def load_dataset(img_size,begin,end):
    image_names = os.listdir("datasets")
    images = []
    image_names = image_names[begin:end]
    for image_name in image_names:
        image = cv.imread("datasets/"+image_name)
        width = image.shape[0]
        height = image.shape[1]
        max_size = min(width,height)
        image = image[(width-max_size)//2:(width+max_size)//2,(height-max_size)//2:(height+max_size)//2]
        image = cv.resize(image,(img_size,img_size))
        image = image.astype(np.float32)#转换数据类型
        image = image/127.5 - 1         #将RGB归一化到0-1范围
        image = image.transpose(2,0,1)  #将图片的channel和xy坐标交换，这一步是为了Self Attention层计算方便
        images.append(image)
    images = np.array(images,np.float32)
    return images

def output_images(prefix,images):
    i = 0
    for image in images:
        cv.imwrite("results/"+prefix+str(i)+".jpg",image)
        i = i + 1