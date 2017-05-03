import numpy
import os
import cv2

image_dir = './images'
buf_data = []

files = os.listdir(image_dir)
for file_name in files:
    im = cv2.imread(os.path.join(image_dir, file_name))
    #buf_data.append(im.reshape(-1))
    buf_data.append(im)
    print(file_name)

sample = len(buf_data[0][0][0])
print(sample)
