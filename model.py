# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:27:54 2020

@author: CupakabraNo1
"""

#MODEL1

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
import constants as const

model = Sequential(name = 'HANDWRITING RECOGNITION')
optimizer = SGD(lr=0.01,momentum=0.9)

model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', input_shape=(28,28,1), name = 'CONV1'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2), name = 'MP1'))
model.add(Conv2D( 64, (3, 3), activation='relu', name = 'CON2'))
model.add(MaxPooling2D(pool_size=(2, 2), name = 'MP2'))
model.add(Dropout(0.25, name = "DROP1"))
model.add(Flatten(name = 'FLATTEN'))

model.add(Dense(256, activation='relu', name = 'INPUT'))
model.add(Dropout(0.25, name = "DROP2"))

model.add(Dense(128, activation='relu', name = 'HIDDEN'))
model.add(Dropout(0.25, name = "DROP3"))

model.add(Dense(const.CLASS_NUMBER, activation='softmax', name = 'OUTPUT'))

model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
