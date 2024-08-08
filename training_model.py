#%%Libraries

from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os 
from short_functions import check_cuda, take_label_area_and_save, prs


check_cuda()


#%% Training model 

#load model
model = YOLO("yolov8n.pt")

#train model
results = model.train(data="data.yaml", epochs=100, imgsz=192)