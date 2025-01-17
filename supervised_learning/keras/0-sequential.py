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
    model.add(K.layers.Dropout(rate=1 - keep_prob))

    for units, activation in zip(layers[1:], activations[1:]):
        model.add(K.layers.Dense(
            units=units, activation=activation,
            kernel_regularizer=K.regularizers.L2(l2=lambtha)
            ))
        model.add(K.layers.Dropout(rate=1 - keep_prob))
    return model
