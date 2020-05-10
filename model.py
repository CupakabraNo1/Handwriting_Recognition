# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:27:54 2020

@author: CupakabraNo1
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU, Lambda
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
import constants as const

model = Sequential(name='HANDWRITING RECOGNITION')
optimizer = SGD(lr=0.1)
input_shape = (28, 28, 1)
filter1 = 32
filter2 = 2*filter1
pool_size = (2, 2)
kernel_size = (3, 3)

model.add(Conv2D(32, kernel_size=kernel_size, activation='relu',
                 input_shape=input_shape, name='CONV1'))
model.add(Conv2D(32, kernel_size=kernel_size, activation='relu', name='CONV2'))
model.add(MaxPooling2D(pool_size=pool_size, name='MP1'))

model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', name='CON3'))
model.add(Conv2D(64, kernel_size=kernel_size, activation='relu', name='CONV4'))
model.add(MaxPooling2D(pool_size=pool_size, name='MP2'))

model.add(Dropout(0.25, name="DROP1"))

model.add(Flatten(name='FLATTEN'))

model.add(Dense(512, name='HIDDEN'))
model.add(LeakyReLU(name='LEAKY5'))
model.add(Dropout(0.2, name="DROP2"))
model.add(Dense(const.CLASS_NUMBER, activation='softmax', name='OUTPUT'))

model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
