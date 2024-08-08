#%%Libraries

from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os 
from my_functions import check_cuda, take_label_area_and_save, prs,get_label_img, img_pro, img_to_text, sample

#%% Check CUDA
check_cuda()

#%% 

model_path = r"runs/detect/train/weights/best.pt"
image_path = 'images.png'
tesseract_path = r"cropped/Tesseract-OCR/tesseract.exe"

plate = get_label_img(model_path,image_path)
plate = img_pro(plate)
plate = cv2.cvtColor(plate,cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(plate)
plt.show()



#%%