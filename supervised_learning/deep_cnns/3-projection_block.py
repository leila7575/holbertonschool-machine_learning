#!/usr/bin/env python3
"""Contains the function projection_block."""


from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block."""
    F11, F3, F12 = filters

    conv2d = K.layers.Conv2D(
        F11,
        (1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)

    batch_normalization = K.layers.BatchNormalization(axis=-1)(conv2d)

    activation = K.layers.Activation('relu')(batch_normalization)

    conv2d_1 = K.layers.Conv2D(
        F3,
        (3, 3),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(activation)

    batch_normalization_1 = K.layers.BatchNormalization(axis=-1)(conv2d_1)

    activation_1 = K.layers.Activation('relu')(batch_normalization_1)

    conv2d_2 = K.layers.Conv2D(
        F12,
        (1, 1),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(activation_1)

    batch_normalization_2 = K.layers.BatchNormalization(axis=-1)(conv2d_2)

    conv2d_3 = K.layers.Conv2D(
        F12,
        (1, 1),
        strides=(s, s),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(A_prev)

    batch_normalization_3 = K.layers.BatchNormalization(axis=-1)(conv2d_3)

    add = K.layers.Add()([batch_normalization_2, batch_normalization_3])

    activation_2 = K.layers.Activation('relu')(add)

    return activation_2
