Preparing Data
--- When training yolov8, I used 400-500 train images. I was have 250 images and I took random images from 250 images, I did image processing on this dataset and I created gray_images and rotated images.
-- Your initial training data should include images taken from different angles and directions – sometimes a close-up of the license plate, sometimes from a distant location.

Preparing Training 
1-- First of all you have download required packages
2-- Create train and test folder. Train images will saved in train folder with labels. Don't create a label folder in train folder. Save images and their labels same path. Do same thing for test data and test folder.
3-- We will train the yolov8 model on cuda so we have to install CUDA.

Warnings !!!
---I didn't load the runs/detect/train/weights file to Lisense Plate Recognition folder, After you train your model with the model_training.py file's codes. You will see your own weights.
---Additionally, I cannot share my training dataset here because I collected it manually from a commercially used website. It may have legal obligations 