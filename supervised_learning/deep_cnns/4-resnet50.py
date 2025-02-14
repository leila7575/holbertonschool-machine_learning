#!/usr/bin/env python3
"""Contains the function resnet50."""


from tensorflow import keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds a ResNet-50 architecture."""

    input_1 = K.layers.Input(shape=((224, 224, 3)))

    conv1 = K.layers.Conv2D(
        64,
        (7, 7),
        padding='same',
        strides=(2, 2),
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(input_1)

    batch_normalization = K.layers.BatchNormalization(axis=-1)(conv1)

    activation = K.layers.Activation('relu')(batch_normalization)

    max_pooling2d = K.layers.MaxPooling2D(
        (3, 3), strides=(2, 2), padding='same'
    )(activation)

    conv2_1 = projection_block(activation, filters=(64, 64, 256))
    conv2_2 = identity_block(conv2_1, filters=(64, 64, 256))
    conv2_3 = identity_block(conv2_2, filters=(64, 64, 256))

    conv3_1 = projection_block(conv2_3, filters=(128, 128, 512))
    conv3_2 = identity_block(conv3_1, filters=(128, 128, 512))
    conv3_3 = identity_block(conv3_2, filters=(128, 128, 512))
    conv3_4 = identity_block(conv3_3, filters=(128, 128, 512))

    conv4_1 = projection_block(conv3_4, filters=(256, 256, 1024))
    conv4_2 = identity_block(conv4_1, filters=(256, 256, 1024))
    conv4_3 = identity_block(conv4_2, filters=(256, 256, 1024))
    conv4_4 = identity_block(conv4_3, filters=(256, 256, 1024))
    conv4_5 = identity_block(conv4_4, filters=(256, 256, 1024))
    conv4_6 = identity_block(conv4_5, filters=(256, 256, 1024))

    conv5_1 = projection_block(conv4_6, filters=(512, 512, 2048))
    conv5_2 = identity_block(conv5_1, filters=(512, 512, 2048))
    conv5_3 = identity_block(conv5_2, filters=(512, 512, 2048))

    average_pooling2d = K.layers.AveragePooling2D()(conv5_3)

    dense = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=K.initializers.HeNormal(seed=0)
    )(average_pooling2d)

    model = K.models.Model(inputs=input_1, outputs=dense)

    return model
