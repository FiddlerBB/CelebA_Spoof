import cv2
import os
import glob
import pandas as pd
import cv2
df = pd.read_csv('data.csv')
for idx, row in df.iterrows():

    img = cv2.imread(row['image_path'])[:,:,::-1]
    real_h, real_w = img.shape[:2]

    with open(row['label_path'], 'r') as f:
        material = f.readline()
        x,y,w,h,score = material.strip().split(' ')
        w = int(float(w))
        h = int(float(h))
        x = int(float(x))
        y = int(float(y))
        w = int(w * (real_w / 224))
        h = int(h * (real_h / 224))
        x = int(x * (real_w / 224))
        y = int(y * (real_h / 224))

        # Crop face based on its bounding box
        y1 = 0 if y < 0 else y
        x1 = 0 if x < 0 else x
        y2 = real_h if y1 + h > real_h else y + h
        x2 = real_w if x1 + w > real_w else x + w

    crop_img = img[y1:y2, x1:x2, :]

    cv2.imwrite('Data/data_face/' + str(row['image_id'])+ '.png', crop_img[:,:,::-1] )







