# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:47:17 2020

@author: CupakabraNo1
"""
import gzip
import numpy as np
import matplotlib.pyplot as plt
import struct
import constants as const

#< TRAINGING SET LOADING >#
f = gzip.open(const.TRAIN_SET_IMAGES,'r')
f.read(16)

buf = f.read(const.IMAGE_SIZE * const.IMAGE_SIZE * const.TRAIN_DATA)
frombuff = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
train_image_data = frombuff.reshape(const.TRAIN_DATA, const.IMAGE_SIZE, const.IMAGE_SIZE, 1)

f = gzip.open(const.TRAIN_SET_LABELS,'r')
f.read(8)
buf = f.read(const.TRAIN_DATA)
buff = np.frombuffer(buf, dtype=np.uint8).astype(np.int)
train_label_data = buff.reshape(const.TRAIN_DATA,1) 

#< TEST SET LOADING >#

f = gzip.open(const.TEST_SET_IMAGES,'r')
f.read(16)

buf = f.read(const.IMAGE_SIZE * const.IMAGE_SIZE * const.TEST_DATA)
frombuff = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
test_image_data = frombuff.reshape(const.TEST_DATA, const.IMAGE_SIZE, const.IMAGE_SIZE,1 )
   
f = gzip.open(const.TEST_SET_LABELS,'r')
f.read(8)
buf = f.read(const.TEST_DATA)
buff = np.frombuffer(buf, dtype=np.uint8).astype(np.int)
test_label_data = buff.reshape(const.TEST_DATA, 1)


#0 #A #K #U #e #o #y
#1 #B #L #V #f #p #z
#2 #C #M #W #g #q
#3 #D #N #X #h #r
#4 #E #O #Y #i #s
#5 #F #P #Z #j #t
#6 #G #Q #a #k #u
#7 #H #R #b #l #v
#8 #I #S #c #m #w
#9 #J #T #d #n #x

