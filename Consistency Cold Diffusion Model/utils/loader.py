import os
import cv2 as cv
import numpy as np

#从数据集中加载所有图片
def load_dataset(img_size):
    image_names = os.listdir("/home/main/dev/diffusionModel/datasets")
    images = []
    for image_name in image_names:
        image = cv.imread("/home/main/dev/diffusionModel/datasets/"+image_name)
        image = cv.resize(image,(img_size,img_size))
        hsv = cv.cvtColor(image,cv.COLOR_BGR2HSV)
        hist = cv.calcHist([hsv],[0],None,[64],[0,64])
        if np.max(hist)<1000 or np.var(hsv[:,:,0])>70:
            images.append(image)
    images = np.array(images)
    return images

def output_images(images):
    i = 0
    for image in images:
        cv.imwrite("results/"+str(i)+".jpg",image)
        i = i + 1