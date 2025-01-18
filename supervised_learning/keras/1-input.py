#!/usr/bin/env python3
"""This module contains the function build_model."""


import tensorflow as tf
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network."""
    inputs = K.Input(shape=(nx, ))

    dense = K.layers.Dense(
        units=layers[0], activation=activations[0],
        kernel_regularizer=K.regularizers.L2(l2=lambtha), input_shape=(nx, )
        )

    x = dense(inputs)

    if len(layers) > 1:
        x = K.layers.Dropout(rate=1 - keep_prob)(x)

    for i, (units, activation) in enumerate(zip(layers[1:], activations[1:])):
        if i < len(layers) - 2:
            x = K.layers.Dense(
                units=units, activation=activation,
                kernel_regularizer=K.regularizers.L2(l2=lambtha)
                )(x)
            x = K.layers.Dropout(rate=1 - keep_prob)(x)
        else:
            x = K.layers.Dense(
                units=units, activation=activation,
                kernel_regularizer=K.regularizers.L2(l2=lambtha)
                )(x)
    model = K.Model(inputs=inputs, outputs=x)

    return model
