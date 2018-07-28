from keras import backend as K
from keras.optimizers import RMSprop, Adam, SGD

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. / (dice_coef(y_true, y_pred) + smooth)


def optimizer(para='adam', lr=1e-3, decay=0):
    if para == 'rmsprop':
        opt = RMSprop(lr=lr, decay=float(decay))
    elif para == 'adam':
        opt = Adam(lr=lr, decay=float(decay))
    elif para == 'amsgrad':
        opt = Adam(lr=lr, decay=float(decay), amsgrad=True)
    elif para == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, nesterov=True, decay=float(decay))

    return opt