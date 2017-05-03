import os
import numpy as np

# using ImageMagick
#
# convert test.jpg -geometry 540x540  -mattecolor "#ffffff" -frame 90x0 test3.jpg

image_size = '540x540'
frame_size = '90x0'
image_dir = './images'

if os.path.exists(image_dir) == False:
    os.mkdir(image_dir)

files = os.listdir(image_dir)
for file_name in files:
    image_path = "{}/{}".format(image_dir, file_name)
    cmd = "convert {} -geometry {}  -mattecolor \"#ffffff\" -frame {} {}".format(image_path, image_size, frame_size, image_path)
    os.system(cmd)
    print(file_name)
