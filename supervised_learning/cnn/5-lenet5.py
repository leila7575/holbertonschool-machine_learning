#!/usr/bin/env python3
"""Contains the function lenet5."""


from tensorflow import keras as K


def lenet5(X):
    """Builds a lenet5 convolutional network."""
    conv1 = K.layers.Conv2D(
        6,
        (5, 5),
        activation='relu',
        padding='same',
        kernel_initializer=K.initializers.HeNormal(seed=None),
        input_shape=(28, 28, 1)
    )(X)
    pool1 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(
        16,
        (5, 5),
        activation='relu',
        padding='valid',
        kernel_initializer=K.initializers.HeNormal(seed=None)
    )(pool1)
    pool2 = K.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(
        120,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=None)
    )(flatten)
    dense2 = K.layers.Dense(
        84,
        activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=None)
    )(dense1)
    dense3 = K.layers.Dense(
        10,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=None)
    )(dense2)
    model = K.Model(inputs=X, outputs=dense3)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
