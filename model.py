# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:27:54 2020

@author: CupakabraNo1
"""

#MODEL1

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import constants as C


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(0.25))

classifier.add(Flatten())

classifier.add(Dense(512, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(144, activation = 'relu'))
classifier.add(Dropout(0.25))
classifier.add(Dense(C.CLASS_NUMBER, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(144, activation = 'relu'))
model.add(Dense(144, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(62, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])