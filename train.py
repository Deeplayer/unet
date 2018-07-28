# -*- coding: utf-8 -*-
import os, cv2, sys, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow as tf

from U_Net import unet
from metrics import dice_coef, optimizer
from InceptionUnet import inception_resnet_v2_fpn


# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'F:/DCB2018/train/'
TEST_PATH = 'F:/DCB2018/stage1_test/'


# Get train and test IDs (folder name)
train_ids = next(os.walk(TRAIN_PATH))[1]
# test_ids = next(os.walk(TEST_PATH))[1]
# print(len(train_ids), len(test_ids))
print(train_ids)

''' np.squeeze(): 从数组的形状中删除单维度条目，即把shape中为1的维度去掉 '''

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)


for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id
    img = imread(path + '/images/' + id + '.png')[:,:,:IMG_CHANNELS]       # (h, w, c)
    # plt.subplot(131)
    # imshow(img)
    #img = cv2.addWeighted(img, 3, cv2.GaussianBlur(img, (0, 0), 11), -3, 128)
    #print(np.max(img))

    # plt.subplot(132)
    # imshow(img)

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
    # plt.show()
    Y_train[n] = mask

print(X_train.shape)
print(Y_train.shape)

X_train /= 255.
X_train -= 0.5
X_train *= 2.


''' Fit model '''
#model = unet(optimizer=optimizer('adam', lr=1e-4))
model = inception_resnet_v2_fpn((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

model.compile(optimizer=optimizer('adam', lr=1e-4), loss='binary_crossentropy', metrics=[dice_coef])

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-2.h5', monitor='val_loss', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
                    callbacks=[earlystopper, checkpointer], verbose=2)


