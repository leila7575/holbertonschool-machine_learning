#!/usr/bin/env python3
"""Contains the function inception_network()."""


from tensorflow import keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds a def GoogLeNet architecture."""

    input_1 = K.layers.Input(shape=((224, 224, 3)))

    conv2d = K.layers.Conv2D(
        64, (7, 7), padding='same', strides=(2, 2), activation='relu'
    )(input_1)

    max_pooling2d = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(conv2d)

    conv2d_1 = K.layers.Conv2D(
        64, (1, 1), padding='same', strides=(1, 1), activation='relu'
    )(max_pooling2d)
    conv2d_2 = K.layers.Conv2D(
        192, (3, 3), padding='same', strides=(1, 1), activation='relu'
    )(conv2d_1)

    max_pooling2d_1 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(conv2d_2)

    inception_3a = inception_block(
        max_pooling2d_1,
        filters=(64, 96, 128, 16, 32, 32)
    )

    inception_3b = inception_block(
        inception_3a,
        filters=(128, 128, 192, 32, 96, 64)
    )

    max_pooling2d_4 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(inception_3b)

    inception_4a = inception_block(
        max_pooling2d_4,
        filters=(192, 96, 208, 16, 48, 64)
    )

    inception_4b = inception_block(
        inception_4a,
        filters=(160, 112, 224, 24, 64, 64)
    )

    inception_4c = inception_block(
        inception_4b,
        filters=(128, 128, 256, 24, 64, 64)
    )

    inception_4d = inception_block(
        inception_4c,
        filters=(112, 144, 288, 32, 64, 64)
    )

    inception_4e = inception_block(
        inception_4d,
        filters=(256, 160, 320, 32, 128, 128)
    )

    max_pooling2d_10 = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(inception_4e)

    inception_5a = inception_block(
        max_pooling2d_10,
        filters=(256, 160, 320, 32, 128, 128)
    )

    inception_5b = inception_block(
        inception_5a,
        filters=(384, 192, 384, 48, 128, 128)
    )

    average_pooling2d = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(inception_5b)

    dropout = K.layers.Dropout(0.4)(average_pooling2d)

    dense = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(dropout)

    model = K.models.Model(inputs=input_1, outputs=dense)

    return model
