#!/usr/bin/env python3
"""Contains autencoder function"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder"""
    input = keras.layers.Input(shape=(input_dims,))
    encoder_layer = input
    for i in hidden_layers:
        encoder_layer = keras.layers.Dense(i, activation='relu')(encoder_layer)
    encoder_output = keras.layers.Dense(
        latent_dims, activation='relu'
    )(encoder_layer)

    encoder = keras.models.Model(input, encoder_output)

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_layer = decoder_input
    for i in (reversed(hidden_layers)):
        decoder_layer = keras.layers.Dense(i, activation='relu')(decoder_layer)
    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(decoder_layer)

    decoder = keras.models.Model(decoder_input, decoder_output)
    output = decoder(encoder(input))
    auto = keras.models.Model(input, output)

    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
