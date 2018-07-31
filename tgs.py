# -*- coding: utf-8 -*-
import os, cv2, sys, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

from InceptionUnet import inception_resnet_v2_fpn

from metrics import dice_bce_loss, dice_coef, optimizer


# Set some parameters
IMG_WIDTH = 139
IMG_HEIGHT = 139
IMG_CHANNELS = 1
# TRAIN_IMG_PATH = 'F:/PythonProjects/deepcare/images'
# TRAIN_MASK_PATH = 'F:/PythonProjects/deepcare/masks'
# TEST_PATH = 'F:/PythonProjects/deepcare/images_test'
TRAIN_IMG_PATH = '../unet/images'
TRAIN_MASK_PATH = '../unet/masks'
TEST_PATH = '../unet/images_test'

train_ids = next(os.walk(TRAIN_IMG_PATH))[2]
test_ids = next(os.walk(TEST_PATH))[2]
print(len(train_ids), len(test_ids))

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

# prepare training set
for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = cv2.imread(TRAIN_IMG_PATH + '/' + id, 0)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    X_train[n] = np.expand_dims(img, axis=-1)

    mask = cv2.imread(TRAIN_MASK_PATH + '/' + id, 0)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    Y_train[n] = np.expand_dims(mask, axis=-1)/255.

# prepare test set
sizes_test = []
for n, id in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = cv2.imread(TEST_PATH + '/' + id, 0)
    sizes_test.append([img.shape[0], img.shape[1]])
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    X_test[n] = np.expand_dims(img, axis=-1)

# normalize images
X_train /= 255.
X_train -= 0.5
X_train *= 2.

X_test /= 255.
X_test -= 0.5
X_test *= 2.


''' Fit model '''
model = inception_resnet_v2_fpn((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#
model.compile(optimizer=optimizer('adam', lr=1e-4), loss=dice_bce_loss, metrics=[dice_coef])
#model.compile(optimizer=optimizer('adam', lr=1e-3), loss=dice_bce_loss, metrics=[dice_coef])
#
earlystopper = EarlyStopping(patience=20, verbose=1)
checkpointer = ModelCheckpoint('../unet/model-tgs-1.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=16, epochs=40,
                    callbacks=[earlystopper, checkpointer], verbose=2)


# Predict on test
model = load_model('../unet/model-tgs-1.h5', custom_objects={'dice_coef': dice_coef})
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
# preds_train_t = (preds_train >= 0.5).astype(np.uint8)
# preds_val_t = (preds_val >= 0.5).astype(np.uint8)
preds_test_t = (preds_test >= 0.5).astype(np.uint8)


# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(cv2.resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1])))


def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm_notebook(enumerate(test_ids))}

sub = pd.DataFrame.from_dict(pred_dict, orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('../unet/submission.csv')


