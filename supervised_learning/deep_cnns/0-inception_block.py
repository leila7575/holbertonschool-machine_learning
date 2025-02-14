#!/usr/bin/env python3
"""Contains the function inception_block."""


from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an inception block."""
    F1, F3R, F3, F5R, F5, FPP = filters

    conv2d = K.layers.Conv2D(
        F1, (1, 1), padding='same', activation='relu'
    )(A_prev)

    conv2d_1 = K.layers.Conv2D(
        F3R, (1, 1), padding='same', activation='relu'
    )(A_prev)
    conv2d_2 = K.layers.Conv2D(
        F3, (3, 3), padding='same', activation='relu'
    )(conv2d_1)

    conv2d_3 = K.layers.Conv2D(
        F5R, (1, 1), padding='same', activation='relu'
    )(A_prev)
    conv2d_4 = K.layers.Conv2D(
        F5, (5, 5), padding='same', activation='relu'
    )(conv2d_3)

    max_pooling2d = K.layers.MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same'
    )(A_prev)
    conv2d_5 = K.layers.Conv2D(
        FPP, (1, 1), padding='same', activation='relu'
    )(max_pooling2d)

    concatenate = K.layers.Concatenate(name='concatenate')(
        [conv2d, conv2d_2, conv2d_4, conv2d_5]
    )

    return concatenate
