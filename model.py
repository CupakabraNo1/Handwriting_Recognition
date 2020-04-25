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

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(const.CLASS_NUMBER, activation='softmax'))

model.compile(optimizer =  SGD(lr=0.01,momentum=0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])