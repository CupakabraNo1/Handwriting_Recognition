# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:09:08 2020

@author: CupakabraNo1
"""

import dataset_loader as dl
from model import model
import constants as const
import keras
from accuracy import AccuracyHistory

history = AccuracyHistory()

model.fit(dl.train_image_data, dl.train_label_data, batch_size=const.BATCH_SIZE, epochs=const.EPOCHS,
          verbose=1, validation_data=(dl.test_image_data, dl.test_label_data), callbacks=[history])
