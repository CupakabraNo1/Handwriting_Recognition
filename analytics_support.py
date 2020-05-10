# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:14:08 2020

@author: CupakabraNo1
"""
# IMPORTS FOR CASE WHERE YOU DIDNT EXECUTE handewriting_recognition_main.py FIRST
# (DO NOT IMPORT THEM IF YOU ALREDY TRAINED YOUR MODEL)
# --------------------------------------------------------
from handwriting_recoginition_main import model, history
import constants as const
import dataset_loader as dl
# --------------------------------------------------------
import matplotlib.pyplot as plt

model.summary()

plt.plot(range(1, const.EPOCHS+1), history.acc, 'green', label='train')
plt.plot(range(1, const.EPOCHS+1), history.val_acc, 'red', label='test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(1, const.EPOCHS+1), history.loss, 'green', label='train')
plt.plot(range(1, const.EPOCHS+1), history.val_loss, 'red', label='test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

score = model.evaluate(dl.test_image_data, dl.test_label_data, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
