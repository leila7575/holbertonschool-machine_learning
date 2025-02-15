#!/usr/bin/env python3
"""Contains the function transition_block."""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer."""

    batch_normalization = K.layers.BatchNormalization(axis=-1)(X)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv1x1 = K.layers.Conv2D(
        (int(nb_filters * compression)),
        (1, 1),
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(activation)

    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(conv1x1)

    return average_pooling2d, nb_filters
