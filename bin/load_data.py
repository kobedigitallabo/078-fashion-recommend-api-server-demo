# -*- coding: utf-8 -*-
import os
from glob import glob
from PIL import Image
import numpy as np

def load_data(target_data=None):
    image_dir = None
    if (target_data == 'training'):
        image_dir = './images_train/'
    elif (target_data == 'test'):
        image_dir = './images_validate/'
    else:
        raise(TypeError('Illigale specified "target_data"'))
    img_shape = (32, 32, 3)
    buf_data = []
    print("Loading {} data...".format(target_data))
    dirs = [os.path.relpath(x, image_dir) for x in glob(os.path.join(image_dir, '*'))]
    # print(dirs)
    for d in dirs:
        for file_path in glob(os.path.join(image_dir, d, '*')):
            # print(file_path)
            im = np.array(Image.open(file_path))
            if im.shape == (img_shape[0], img_shape[1], img_shape[2]):
                buf_data.append([im, int(d)])

    np.random.shuffle(buf_data)

    x_buf = []
    y_buf = []
    for d in buf_data:
        x_buf.append(d[0])
        y_buf.append(d[1])
    x = np.array(x_buf, dtype=np.float32)
    y = np.array(y_buf, dtype=np.uint8)

    print("Loaded.")
    print("Number of {} data: {}".format(target_data, len(x)))
    print("Number of {} labels : {}".format(target_data, len(y)))

    return x, y

if __name__ == '__main__':
    x_train, y_train = load_data(target_data='training')
    x_test, y_test = load_data(target_data='test')
