# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:09:08 2020

@author: CupakabraNo1
"""

import dataset_loader as dl
from model import classifier,model
import constants as C

model.fit(dl.train_image_data,
        dl.train_label_data,
        #batch_size=C.BATCH_SIZE,
        epochs=5,
        validation_data=(dl.test_image_data,dl.test_label_data))

#score = classifier.evaluate(dl.test_image_data,dl.test_label_data, verbose=0)
score = model.evaluate(dl.test_image_data,dl.test_label_data, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])