#!/usr/bin/env python3
"""Contains the function dense_block."""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block."""

    for i in range(layers):
        batch_normalization_1 = K.layers.BatchNormalization(axis=-1)(X)
        activation_1 = K.layers.Activation('relu')(batch_normalization_1)
        conv1x1 = K.layers.Conv2D(
            (4 * growth_rate),
            (1, 1),
            padding='same',
            kernel_initializer=K.initializers.HeNormal(seed=0)
        )(activation_1)

        batch_normalization_2 = K.layers.BatchNormalization(axis=-1)(conv1x1)
        activation_2 = K.layers.Activation('relu')(batch_normalization_2)
        conv3x3 = K.layers.Conv2D(
            growth_rate,
            (3, 3),
            padding='same',
            kernel_initializer=K.initializers.HeNormal(seed=0)
        )(activation_2)

        X = K.layers.concatenate([X, conv3x3])
        nb_filters += growth_rate

    return X, nb_filters
