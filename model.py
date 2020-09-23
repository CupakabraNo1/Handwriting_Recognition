# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:27:54 2020

@author: CupakabraNo1
"""

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from keras.optimizers import SGD, Adam
import constants as const

model = Sequential(name='HANDWRITING RECOGNITION')

optimizer1 = SGD(lr=0.1)
optimizer2 = Adam()


filter1 = 32
filter2 = 2 * filter1

pool_size = (2, 2)
kernel_size = (3, 3)

model.add(Conv2D(filter1, kernel_size = kernel_size, activation = 'relu', input_shape = (const.IMAGE_SIZE, const.IMAGE_SIZE, 1), name = 'CONV1'))
model.add(Conv2D(filter1, kernel_size = kernel_size, activation = 'relu', name = 'CONV2'))
model.add(Conv2D(filter1, kernel_size = kernel_size, activation = 'relu', name = 'CONV3'))
model.add(MaxPooling2D(pool_size = pool_size, name = 'MP1'))

model.add(Conv2D(filter2, kernel_size = kernel_size, activation = 'relu', name = 'CONV4'))
model.add(Conv2D(filter2, kernel_size = kernel_size, activation = 'relu', name = 'CONV5'))
model.add(Conv2D(filter2, kernel_size = kernel_size, activation = 'relu', name = 'CONV6'))
model.add(MaxPooling2D(pool_size = pool_size, name = 'MP2'))

#model.add(Dropout(0.25, name = "DROP1"))

model.add(Flatten(name = 'FLATTEN'))

model.add(Dense(512, name = 'HIDDEN1', activation = 'relu'))
model.add(LeakyReLU(name = 'LEAKY5'))
#model.add(Dropout(0.2, name="DROP2"))
model.add(Dense(const.CLASS_NUMBER, activation='softmax', name='OUTPUT'))

model.compile(optimizer = optimizer1, loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()
