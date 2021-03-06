# -*- coding: utf-8 -*-
TRAIN_SET_IMAGES = 'emnist-byclass/train-images-idx3-ubyte.gz'
TRAIN_SET_LABELS = 'emnist-byclass/train-labels-idx1-ubyte.gz'
TEST_SET_IMAGES = 'emnist-byclass/test-images-idx3-ubyte.gz'
TEST_SET_LABELS = 'emnist-byclass/test-labels-idx1-ubyte.gz'

IMAGE_SIZE = 28
TRAIN_DATA = 697932
# 697932
TEST_DATA = 116323
# 116323

LABEL_ENCODING = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S','T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


def decode(position):
    return LABEL_ENCODING[position]


EPOCHS = 15
BATCH_SIZE = 256
CLASS_NUMBER = 62

SIZE_FOR_ANALISIS = 1000
ACCTIVATION_BOUNDARY = 2.0

EXT_CSV = '.xlsm'
LAYERS_FOR_ACT = ['FLATTEN', 'HIDDEN1']
START_OF_EXCEL = 2



