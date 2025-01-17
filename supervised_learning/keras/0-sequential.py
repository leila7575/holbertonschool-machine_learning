#!/usr/bin/env python3
"""This module contains the function build_model."""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import layers, regularizers


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network."""
    model = Sequential()

    model.add(Dense(
        units=layers[0], activation=activations[0],
        kernel_regularizer=regularizers.L2(l2=lambtha), input_shape=(nx, )
        ))
    model.add(Dropout(rate=1 - keep_prob))

    for units, activation in zip(layers[1:], activations[1:]):
        model.add(Dense(
            units=units, activation=activation,
            kernel_regularizer=regularizers.L2(l2=lambtha)
            ))
        model.add(Dropout(rate=1 - keep_prob))
    return model
