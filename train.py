
import os, cv2
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

from U_Net import unet
from InceptionUnet import inception_unet


# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = '../unet/train/'
TEST_PATH = '../unet/stage1_test/'


# Get train and test IDs (folder name)
train_ids = next(os.walk(TRAIN_PATH))[1]
# test_ids = next(os.walk(TEST_PATH))[1]
# print(len(train_ids), len(test_ids))
print(train_ids)


# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id
    img = imread(path + '/images/' + id + '.png')[:,:,:IMG_CHANNELS]       # (h, w, c)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 11), -4, 128)
    #print(np.max(img))
    # plt.subplot(131)
    # imshow(img)
    # plt.subplot(132)
    # imshow(img_1)

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for m in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + m)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True),
                               axis=-1)
        mask = np.maximum(mask_, mask)

    # plt.subplot(133)
    # imshow(np.squeeze(mask))
    #plt.show()
    Y_train[n] = mask

print(X_train.shape)
print(Y_train.shape)

X_train /= 255.
X_train -= 0.5
X_train *= 2.


''' Fit model '''
model = unet()
#model = inception_unet(Adam(2e-4), IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-2.h5', monitor='val_acc', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
                    callbacks=[earlystopper, checkpointer], verbose=2)


