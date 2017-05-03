import base64
from io import BytesIO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import json
import urllib
import re
from keras.models import model_from_json
from keras import backend as K

class Recommender():

    def __init__(self):
        self.image_dir = './images/'
        self.model_path = './report/5353-1038-epoch20-1492244888.443806/model-5353-1038-epoch20-1492244888.443806.json'
        self.weights_path = './report/5353-1038-epoch20-1492244888.443806/weights-5353-1038-epoch20-1492244888.443806.hdf5'

    # Base64テキストをNumPy配列に変換する
    def __convetB64ToNpa(self, b64txt=None):
        if(b64txt == None): raise ValueError('Data not found')
        b64txt = b64txt.replace(' ', '+')
        file = BytesIO(base64.b64decode(b64txt))
        im = Image.open(file)
        data = np.asarray(im)
        # plt.imshow(data)
        # plt.show()
        # plt.clf()   # plotを初期化
        return data

    # NumPy配列画像データから種別を推測する
    def __getLabelByImage(self, data=None):
        buf = []
        buf.append(data)
        x = np.array(buf, dtype=np.float32)
        x /= 255  # 0-255の値を0-1に変換(Kerasに読み込むために必要)
        model = None
        with open(self.model_path) as f:
            json = f.read()
            model = model_from_json(json)
        # 学習済みモデル(weight含む)のロード
        model.load_weights(self.weights_path)
        score = model.predict(x)
        K.clear_session()
        return np.argmax(score)

    # ラベルからサンプル画像名を取得する
    def __getImageNamesByLabel(self, label=None):
        files = os.listdir(os.path.join(self.image_dir, str(label)))
        images = []
        for file_name in files:
            images.append(file_name)
        random.shuffle(images)
        return images[0:5]

    def reccommend(self, data=None):
        images = None
        label = None
        err = ""
        try:
            sample_data = self.__convetB64ToNpa(data)
            label = self.__getLabelByImage(sample_data)
            images = self.__getImageNamesByLabel(label)
        except Exception as e:
            err = e.args
        ret = {
            "error": err,
            "label": str(label),
            "images": images
        }
        return json.dumps(ret)


if __name__ == '__main__':
    # Base64テキストを読み込んでnumpy配列にする
    b64txt = None
    with open("./fromApi.txt", "r") as f:
        b64txt = f.read()

    Rm = Recommender()
    images = Rm.reccommend(b64txt)

    ret = json.dumps(images)
    print("ret: {}".format(ret))

    # plt.imshow(data)
    # plt.show()
    # plt.clf()   # plotを初期化

"""
# 画像をbase64エンコードして保存する
file = None
with open("./images/122046110901.jpg", "rb") as f:
    file = f.read()
print(base64.b64encode(file))
"""
