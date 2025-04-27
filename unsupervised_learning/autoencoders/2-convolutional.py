#!/usr/bin/env python3
"""Contains convolutional autoencoder function"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates convolutional autoencoder"""
    input = keras.layers.Input(shape=input_dims)

    encoder_layer = input
    for i in filters:
        encoder_layer = keras.layers.Conv2D(
            i, (3, 3), padding='same', activation='relu'
        )(encoder_layer)
        encoder_layer = keras.layers.MaxPooling2D(
            (2, 2), padding='same'
        )(encoder_layer)

    encoder = keras.models.Model(input, encoder_layer)

    decoder_input = keras.layers.Input(shape=latent_dims)

    decoder_layer = decoder_input
    reversed_filters = filters[::-1]
    for i in reversed_filters[:-1]:
        decoder_layer = keras.layers.Conv2D(
            i, (3, 3), padding='same', activation='relu'
        )(decoder_layer)
        decoder_layer = keras.layers.UpSampling2D((2, 2))(decoder_layer)

    decoder_layer = keras.layers.Conv2D(
        reversed_filters[-1], (3, 3), padding='valid', activation='relu'
    )(decoder_layer)
    decoder_layer = keras.layers.UpSampling2D((2, 2))(decoder_layer)
    decoder_output = keras.layers.Conv2D(
        input_dims[2], (3, 3), padding='same', activation='sigmoid'
    )(decoder_layer)

    decoder = keras.models.Model(decoder_input, decoder_output)

    auto_output = decoder(encoder(input))
    auto = keras.models.Model(input, auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
