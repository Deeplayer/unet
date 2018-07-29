# -*- coding: utf-8 -*-
import os, cv2, sys, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf

from Unet import unet
from metrics import dice_coef, dice_coef_loss, optimizer
from InceptionUnet import inception_resnet_v2_fpn


# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'F:/DCB2018/train/'
TEST_PATH = 'F:/DCB2018/stage2_test/'


# Get train and test IDs (folder name)
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
print(len(train_ids), len(test_ids))
#print(train_ids)

''' np.squeeze(): 从数组的形状中删除单维度条目，即把shape中为1的维度去掉 '''

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id
    img = cv2.imread(path + '/images/' + id + '.png')
    #img = imread(path + '/images/' + id + '.png')[:,:,:IMG_CHANNELS]       # (h, w, c)
    # plt.subplot(131)
    # imshow(img)
    img = cv2.addWeighted(img, 3, cv2.GaussianBlur(img, (0, 0), 11), -3, 128)
    #print(np.max(img))

    # plt.subplot(132)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    # imshow(img)

    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for m in next(os.walk(path + '/masks/'))[2]:
        mask_ = cv2.imread(path + '/masks/' + m, 0)
        mask_ = cv2.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask_, mask)

    # plt.subplot(133)
    # imshow(np.squeeze(mask))
    Y_train[n] = mask

print(X_train.shape)
print(Y_train.shape)

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
sizes_test = []
print('Getting and resizing test images ... ')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = cv2.imread(path + '/images/' + id_ + '.png')

    sizes_test.append([img.shape[0], img.shape[1]])
    img = cv2.addWeighted(img, 3, cv2.GaussianBlur(img, (0, 0), 11), -3, 128)
    img = cv2.cvtColor(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)), cv2.COLOR_BGR2RGB)
    X_test[n] = img
    # plt.imshow(img)
    # plt.show()

X_train /= 255.
X_train -= 0.5
X_train *= 2.

X_test /= 255.
X_test -= 0.5
X_test *= 2.


''' Fit model '''
# model = unet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
# #model = inception_resnet_v2_fpn((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
#
# #model.compile(optimizer=optimizer('adam', lr=1e-4), loss='binary_crossentropy', metrics=[dice_coef])
# model.compile(optimizer=optimizer('adam', lr=1e-3), loss='binary_crossentropy', metrics=[dice_coef])
#
# earlystopper = EarlyStopping(patience=5, verbose=1)
# checkpointer = ModelCheckpoint('model-dsbowl2018-2.h5', monitor='val_loss', verbose=1, save_best_only=True)
# results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=30,
#                     callbacks=[earlystopper, checkpointer], verbose=2)


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)

    return K.mean(K.stack(prec), axis=0)


# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'dice_coef': dice_coef, 'dice_coef_loss':dice_coef_loss})
# preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions
# preds_train_t = (preds_train >= 0.5).astype(np.uint8)
# preds_val_t = (preds_val >= 0.5).astype(np.uint8)
preds_test_t = (preds_test >= 0.5).astype(np.uint8)


# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_test_t))
plt.subplot(121)
plt.imshow(X_test[ix])
plt.subplot(122)
plt.imshow(np.squeeze(preds_test_t[ix]))
plt.show()


# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(cv2.resize(np.squeeze(preds_test[i]), (sizes_test[i][0], sizes_test[i][1])))

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
IID = new_test_ids
EP = list(pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x)))

for i in test_ids:
    if i not in new_test_ids:
        IID.append(i)
        EP.append('')

sub['ImageId'] = IID
sub['EncodedPixels'] = EP

sub.to_csv('sub-dsbowl2018-1.csv', index=False)
