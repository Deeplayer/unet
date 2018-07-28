from keras.models import Model
from keras.optimizers import Adam, SGD, Adadelta
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, add, Dropout, \
     Conv2DTranspose, AtrousConvolution2D, BatchNormalization, Lambda, Dense, Flatten
from keras import backend as K


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. / (dice_coef(y_true, y_pred) + smooth)


def BNConvolution2D(filters, ks=(1, 1), strides=(1, 1), padding='same'):
    def f(_input):
        conv = Conv2D(filters=filters, kernel_size=ks, strides=strides, padding=padding, kernel_initializer='he_normal')(_input)
        norm = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(conv)
        return ELU()(norm)

    return f


def BN(_input):
    inputs_norm = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(_input)

    return ELU()(inputs_norm)


def reduction_a(inputs, k=64, l=64, m=96, n=96):
    " 35x35 -> 17x17 "
    inputs_norm = BN(inputs)
    # c1
    pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputs_norm)
    # c2
    conv2 = Conv2D(n, (3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', dilation_rate=(2, 2))(inputs_norm)
    # c3
    conv3_1 = BNConvolution2D(k, (1, 1), strides=(1, 1), padding='same')(inputs_norm)
    conv3_2 = BNConvolution2D(l, (3, 3), strides=(1, 1), padding='same')(conv3_1)
    conv3_2 = Conv2D(m, (3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', dilation_rate=(2, 2))(conv3_2)

    res = concatenate([pool1, conv2, conv3_2], axis=1)

    return res


def reduction_b(inputs):
    " 17x17 -> 8x8 "
    inputs_norm = BN(inputs)
    # c1
    pool1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inputs_norm)
    # c2
    conv2_1 = BNConvolution2D(64, (1, 1), strides=(1, 1), padding='same')(inputs_norm)
    conv2_2 = Conv2D(96, (3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', dilation_rate=(2, 2))(conv2_1)
    # c3
    conv3_1 = BNConvolution2D(64, (1, 1), strides=(1, 1), padding='same')(inputs_norm)
    conv3_2 = Conv2D(72, (3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', dilation_rate=(2, 2))(conv3_1)
    # c4
    conv4_1 = BNConvolution2D(64, (1, 1), strides=(1, 1), padding='same')(inputs_norm)
    conv4_2 = BNConvolution2D(72, (3, 3), strides=(1, 1), padding='same')(conv4_1)
    conv4_3 = Conv2D(80, (3, 3), kernel_initializer='he_normal', strides=(2, 2), padding='same', dilation_rate=(2, 2))(conv4_2)
    # merge
    res = concatenate([pool1, conv2_2, conv3_2, conv4_3], axis=1)

    return res


def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(residual._keras_shape[1], (1, 1), strides=(stride_width, stride_height),
                          kernel_initializer="he_normal", padding="valid", dilation_rate=(2,2))(_input)

    return add([shortcut, residual])


def rblock(inputs, num, depth, scale=0.1):
    residual = Conv2D(depth, (num, num), kernel_initializer='he_normal', padding='same', dilation_rate=(2,2))(inputs)
    residual = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(residual)
    residual = Lambda(lambda x: x*scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res)


def inception_block(inputs, depth, batch_mode, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    c1_1 = Conv2D(int(depth / 4), (1, 1), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(inputs)

    c2_1 = Conv2D(int(depth / 8 * 3), (1, 1), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(c1_1)
    c2_1 = actv()(c2_1)
    if splitted:
        c2_2 = Conv2D(int(depth / 2), (1, 3), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(c2_1)
        c2_2 = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(int(depth / 2), (3, 1), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(c2_2)
    else:
        c2_3 = Conv2D(int(depth / 2), (3, 3), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(c2_1)

    c3_1 = Conv2D(int(depth / 16), (1, 1), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(inputs)
    # missed batch norm
    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Conv2D(int(depth / 8), (1, 5), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(c3_1)
        c3_2 = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(int(depth / 8), (5, 1), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(c3_2)
    else:
        c3_3 = Conv2D(int(depth / 8), (5, 5), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(c3_1)

    p4_1 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    c4_2 = Conv2D(int(depth / 8), (1, 1), kernel_initializer='he_normal', padding='same', dilation_rate=(2, 2))(p4_1)

    res = concatenate([c1_1, c2_3, c3_3, c4_2], axis=3)
    res = BatchNormalization(momentum=0.99, epsilon=0.001, center=True, scale=True)(res)
    res = actv()(res)

    return res


def inception_unet(opt, IMG_ROWS, IMG_COLS, C):
    splitted = False
    act = 'elu'

    " --- encoder --- "
    inputs = Input(shape=(IMG_ROWS, IMG_COLS, C))
    conv1 = inception_block(inputs, 64, batch_mode=2, splitted=splitted, activation=act)
    # conv1 = inception_block(conv1, 32, batch_mode=2, splitted=splitted, activation=act)

    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = BNConvolution2D(32, (3, 3), padding='same', strides=(2, 2))(conv1)
    pool1 = Dropout(0.5)(pool1)

    conv2 = inception_block(pool1, 128, batch_mode=2, splitted=splitted, activation=act)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BNConvolution2D(128, (3, 3), padding='same', strides=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = inception_block(pool2, 256, batch_mode=2, splitted=splitted, activation=act)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BNConvolution2D(256, (3, 3), padding='same', strides=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = inception_block(pool3, 512, batch_mode=2, splitted=splitted, activation=act)
    # print(conv4.shape)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BNConvolution2D(512, (3, 3), padding='same', strides=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    # conv5 = inception_block(conv5, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)
    # print(conv5.shape)

    #
    pre = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', dilation_rate=(2, 2))(conv5)
    pre = Flatten()(pre)
    aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)

    " --- decoder --- "
    after_conv4 = rblock(conv4, 1, 512)
    # print(after_conv4.shape)
    x = UpSampling2D(size=(2, 2))(conv5)
    # print(x.shape)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, splitted=splitted, activation=act)
    conv6 = Dropout(0.5)(conv6)

    after_conv3 = rblock(conv3, 1, 256)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=1)
    conv7 = inception_block(up7, 128, batch_mode=2, splitted=splitted, activation=act)
    conv7 = Dropout(0.5)(conv7)

    after_conv2 = rblock(conv2, 1, 128)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=1)
    conv8 = inception_block(up8, 64, batch_mode=2, splitted=splitted, activation=act)
    conv8 = Dropout(0.5)(conv8)

    after_conv1 = rblock(conv1, 1, 64)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=1)
    conv9 = inception_block(up9, 32, batch_mode=2, splitted=splitted, activation=act)
    # conv9 = inception_block(conv9, 32, batch_mode=2, splitted=splitted, activation=act)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (5, 1), kernel_initializer='he_normal', activation='sigmoid', dilation_rate=(IMG_ROWS, IMG_COLS),
                    name='main_output')(conv9)
    #print (conv10._keras_shape)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[dice_coef])

    return model


# model = inception_unet(Adam(2e-4), 128, 128, 3)
# model.summary()


