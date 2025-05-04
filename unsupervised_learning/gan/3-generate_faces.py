#!/usr/bin/env python3
"""contains convolutional_GenDiscr for building generator
and discriminator networks for convolutional GAN model"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def convolutional_GenDiscr():
    """Buils convolutional GAN generator and discriminator network"""

    def get_generator():
        """Builds generator network."""
        # generator model
        input = keras.layers.Input(shape=(16,))
        dense_layer = keras.layers.Dense(2048, activation='tanh')(input)
        reshape_layer = keras.layers.Reshape((2, 2, 512))(dense_layer)
        filters = [64, 16, 1]
        x = reshape_layer
        for i in filters:
            x = keras.layers.UpSampling2D((2, 2))(x)
            x = keras.layers.Conv2D(i, (3, 3), padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('tanh')(x)

        generator = keras.models.Model(input, x, name='generator')
        return generator

    def get_discriminator():
        """Builds discriminator network"""
        # discriminator model
        input = keras.layers.Input(shape=(16, 16, 1))
        x = input
        filters = [32, 64, 128, 256]
        for i in filters:
            x = keras.layers.Conv2D(i, (3, 3), padding='same')(x)
            x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
            x = keras.layers.Activation('tanh')(x)
        flattened = keras.layers.Flatten()(x)
        output = keras.layers.Dense(1, activation='tanh')(flattened)

        discriminator = keras.models.Model(input, output, name='discriminator')
        return discriminator

    return get_generator(), get_discriminator()
