# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:09:08 2020

@author: CupakabraNo1
"""

import dataset_loader as dl
from model import model
import constants as const
import keras
import matplotlib.pyplot as plt
from keras.callbacks import Callback

class AccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.loss = []
        self.val_acc = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('accuracy'))
        self.loss.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

        

dl.train_image_data = dl.train_image_data.astype('float32')
dl.test_image_data = dl.test_image_data.astype('float32')
dl.train_image_data /= 255
dl.test_image_data /= 255

dl.train_label_data = keras.utils.to_categorical(dl.train_label_data, const.CLASS_NUMBER)
dl.test_label_data = keras.utils.to_categorical(dl.test_label_data, const.CLASS_NUMBER)

history = AccuracyHistory()

model.fit(dl.train_image_data, dl.train_label_data, batch_size=const.BATCH_SIZE, epochs=const.EPOCHS, verbose=1, validation_data=(dl.test_image_data, dl.test_label_data), callbacks=[history])

plt.plot(range(1,7), history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(range(1,7), history.loss)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(range(1,7), history.val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

plt.plot(range(1,7), history.val_loss)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#score = classifier.evaluate(dl.test_image_data,dl.test_label_data, verbose=0)
score = model.evaluate(dl.test_image_data,dl.test_label_data, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 