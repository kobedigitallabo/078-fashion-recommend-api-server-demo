import sys
import os
import json
import numpy as np
from PIL import Image
from keras.models import model_from_json
from keras import backend as K

input_image_label = "1"

img_shape = (32, 32, 3)

model_path = './report/5353-1038-epoch20-1492244888.443806/model-5353-1038-epoch20-1492244888.443806.json'
weights_path = './report/5353-1038-epoch20-1492244888.443806/weights-5353-1038-epoch20-1492244888.443806.hdf5'

image_dir = './images_validate/'

def load_data(label=0):
    buf_data = []
    files = os.listdir(os.path.join(image_dir, str(label)))
    np.random.shuffle(files)
    file_name = files[0]
    print(file_name)
    im = np.array(Image.open(os.path.join(image_dir, str(label), file_name)))
    if im.shape == (img_shape[0], img_shape[1], img_shape[2]):
        buf_data.append(im)
    return np.array(buf_data, dtype=np.float32)

def main():
    x = load_data(input_image_label)   # 画像データをランダムに一つ読み込む
    x /= 255  # 0-255の値を0-1に変換(Kerasに読み込むために必要)
    model = None
    with open(model_path) as f:
        json = f.read()
        model = model_from_json(json)
    # 学習済みモデル(weight含む)のロード
    model.load_weights(weights_path)
    score = model.predict(x)
    K.clear_session()
    return np.argmax(score)

if __name__ == '__main__':
    ret = []
    for x in range(100):    # 100回推測して結果を配列で得る
        ret.append(main())
    print(ret)
    print('finish.')
