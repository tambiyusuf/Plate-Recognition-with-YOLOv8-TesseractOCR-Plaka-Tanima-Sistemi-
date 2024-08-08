from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os 
import pytesseract
import random
from glob import glob
import numpy as np


def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available! Number of GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

def take_label_area_and_save(model, img_path):
    image = cv2.imread(img_path)

    # Predict image with trained model weights(best.pt)
    results = model(image)

    # Sonuçları işleme ve koordinatları alma
    boxes = results[0].boxes.xyxy.cpu().numpy()  # coordinates of label on x_min, y_min, x_max, y_max format
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # class's labels
    
    # check the cropped folder if not exist create folder
    cropped_dir = 'cropped'
    if not os.path.exists(cropped_dir):
        os.makedirs(cropped_dir)

    print("Coordinates:")
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x_min, y_min, x_max, y_max = map(int, box)
        print(f"Box: {(x_min, y_min, x_max, y_max)}, Score: {score:.2f}, Label: {label}")

        # took plate area from label coor.
        cropped_img = image[y_min:y_max, x_min:x_max]

        # Save cropped plate area image to 'cropped' folder
        # split the test image's name
        name, ext = os.path.splitext(img_path)
        # rename cropped image
        new_name = f"{name}_cropped{ext}"
        cropped_path = os.path.join(cropped_dir, new_name)
        cv2.imwrite(cropped_path, cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
    
    result_image = results[0].plot()
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    # show the image and plate together
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('İmage')

    plt.subplot(1, 2, 2)
    plt.imshow(cropped_img)
    plt.axis('off')
    plt.title('Plate')

    plt.show()
    
def get_label_img(model_path,image_path):
    
    model = YOLO(model_path)
    image = cv2.imread(image_path) 
    
    
    results = model(image)

    
    boxes = results[0].boxes.xyxy.cpu().numpy()  
    scores = results[0].boxes.conf.cpu().numpy()  
    labels = results[0].boxes.cls.cpu().numpy()  


    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x_min, y_min, x_max, y_max = map(int, box)

    
        plate = image[y_min:y_max, x_min:x_max]
    
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB) 

    return plate

def img_pro(image):

    enlarged_image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    
    gray_image = cv2.cvtColor(enlarged_image, cv2.COLOR_BGR2GRAY)

    # Do all pixel value 0 if pixel value is under 150  
    gray_image[gray_image < 135] = 0

    # Do all pixel value 255 if pixel value is over 150 
    gray_image[gray_image > 150] = 255

    # sharpening 
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(gray_image, -1, sharpen_kernel)

    final_image = np.where((sharpened_image > 0) & (sharpened_image < 255), 128, sharpened_image)
    final_image = np.where(final_image == 128, 255, final_image)

    return final_image

def prs(image_path, model_path,tesseract_path):
    
    model = YOLO(model_path)
    image = cv2.imread(image_path) 
    
    
    results = model(image)

    # processing result ve and taking coordinates
    boxes = results[0].boxes.xyxy.cpu().numpy()  # coordinates of label box on x_min, y_min, x_max, y_max format
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # Class's labels


    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x_min, y_min, x_max, y_max = map(int, box)

        # take plate label area
        plate = image[y_min:y_max, x_min:x_max]
    
    plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB) 
    
    pytesseract.pytesseract.tesseract_cmd = tesseract_path 


    text = pytesseract.image_to_string(plate)
    print(text)
    
def take_name(folder_path):
    
    # Klasördeki tüm .png dosyalarının yolunu al
    png_files = glob(os.path.join(folder_path, '*.png'))
    
    # Dosya isimlerini al
    filenames = [os.path.basename(file) for file in png_files]
    
    return filenames

def gray_filter(file_path, img_num, output_path):
    
    filenames = take_name(file_path)
    
    selected_names = random.sample(filenames, img_num)
    
    for name in selected_names:
        
        image = cv2.imread(f"train1/{name}")
        
        if image is None:
            print("image couldn't read...")
            continue
        
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # split the image path 
        name, ext = os.path.splitext(name)
        # rename
        new_name = f"{name}_gray{ext}"
        print("new name : ",new_name)
        output_file = os.path.join(output_path, new_name)
        print(output_file)
        
        # check the output file 
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
   
        #     Save the gray image
        if cv2.imwrite(output_file, gray_image):
            print(f"Saved: {output_file}")
        else:
            print(f"Couldn't Save': {output_file}")

def rotate_image(file_path, img_num, output_path, degree):
    
    filenames = take_name(file_path)
    
    selected_names = random.sample(filenames, img_num)
    
    for name in selected_names:
        
        image = cv2.imread(f"train1/{name}")
        
        if image is None:
            print("image couldn't read...")
            continue
        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if degree != 0:
            # take the shape of image
            (h, w) = image.shape[:2]
    
            # create rotation matris
            M = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1.0)
    
            # rotate image
            rotated = cv2.warpAffine(image, M, (w, h))

        # split the image path 
        name, ext = os.path.splitext(name)
        # rename
        new_name = f"{name}_rotated{ext}"
        print("new name : ",new_name)
        output_file = os.path.join(output_path, new_name)
        print(output_file)
        
        # check the output file 
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
   
        #     Save the roated image
        if cv2.imwrite(output_file, rotated):
            print(f"Saved: {output_file}")
        else:
            print(f"Couldn't Save': {output_file}")

def img_to_text(image,tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    text = pytesseract.image_to_string(image)
    print(text)



def sample(image_path, model_path):
    image = cv2.imread(image_path)
    model = YOLO(model_path)
    results = model(image)
    return results






















