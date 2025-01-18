#!/usr/bin/env python3
"""This module contains the function build_model."""


import tensorflow as tf
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network."""
    model = K.models.Sequential()

    model.add(K.layers.Dense(
        units=layers[0], activation=activations[0],
        kernel_regularizer=K.regularizers.L2(l2=lambtha), input_shape=(nx, )
        ))

    if len(layers) > 1:
        model.add(K.layers.Dropout(rate=1 - keep_prob))

    for i, (units, activation) in enumerate(zip(layers[1:], activations[1:])):
        if i < len(layers) - 2:
            model.add(K.layers.Dense(
                units=units, activation=activation,
                kernel_regularizer=K.regularizers.L2(l2=lambtha)
                ))
            model.add(K.layers.Dropout(rate=1 - keep_prob))
        else:
            model.add(K.layers.Dense(
                units=units, activation=activation,
                kernel_regularizer=K.regularizers.L2(l2=lambtha)
                ))
    return model
