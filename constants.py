# -*- coding: utf-8 -*-
TRAIN_SET_IMAGES = 'emnist-byclass/train-images-idx3-ubyte.gz'
TRAIN_SET_LABELS = 'emnist-byclass/train-labels-idx1-ubyte.gz'
TEST_SET_IMAGES = 'emnist-byclass/test-images-idx3-ubyte.gz'
TEST_SET_LABELS = 'emnist-byclass/test-labels-idx1-ubyte.gz'

IMAGE_SIZE = 28
TRAIN_DATA = 690000
#697932
TEST_DATA = 110000
#116323

LABEL_ENCODING = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def decode(position):
    return LABEL_ENCODING[position]

EPOCHS = 6
BATCH_SIZE = 128
CLASS_NUMBER = 62
TEST_PROCENT = 0.2
