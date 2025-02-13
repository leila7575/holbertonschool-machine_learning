#!/usr/bin/env python3
"""Contains the function lenet5."""


import tensorflow.keras as K


def lenet5(X):
    """Builds a lenet5 convolutional network."""
    model = K.models.Sequential([
        K.layers.Conv2D(
            6,
            (5, 5),
            activation='relu',
            padding='same',
            kernel_initializer=K.initializers.HeNormal(seed=None),
            input_shape=(28, 28, 1)
        ),
        K.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        K.layers.Conv2D(
            16,
            (5, 5),
            activation='relu',
            padding='valid',
            kernel_initializer=K.initializers.HeNormal(seed=None)
        ),
        K.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        K.layers.Flatten(),
        K.layers.Dense(
            120,
            activation='relu',
            kernel_initializer=K.initializers.HeNormal(seed=None)
        ),
        K.layers.Dense(
            84,
            activation='relu',
            kernel_initializer=K.initializers.HeNormal(seed=None)
        ),
        K.layers.Dense(
            10,
            activation=None,
            kernel_initializer=K.initializers.HeNormal(seed=None)
        ),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
