# -*- coding: utf-8 -*-
import numpy as np
import os
import keras    # version 2.02
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Activation
from keras import backend as K
from keras.utils import np_utils
from load_data import load_data
from reporter import Reporter
from PIL import Image
import pickle

input_shape = (32, 32, 3)
epoch = 20
batch_size = 30
optimizer='adam'

x_train, y_train = load_data(target_data='training')
x_test, y_test   = load_data(target_data='test')
x_train /= 255
x_test /= 255

nb_classes = len(set(y_test))
print("Class: {}".format(nb_classes))

y_train = np_utils.to_categorical(y_train, len(set(y_test)))
y_test = np_utils.to_categorical(y_test, len(set(y_test)))

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epoch,
                    batch_size=batch_size,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)

# 学習結果を保存する
reporter = Reporter((x_train.shape[0], x_test.shape[0], epoch))
reporter.report(model=model, history=history, show_loss_plot=True)

# https://github.com/tensorflow/tensorflow/issues/3388
K.clear_session()

# スコア、精度の出力
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Finish.')
