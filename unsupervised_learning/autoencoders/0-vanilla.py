#!/usr/bin/env python3
"""Contains autencoder function"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder"""
    input = keras.layers.Input(shape=(input_dims,))
    encoder = keras.layers.Dense(hidden_layers[0], activation='relu')(input)
    for i in hidden_layers[1:]:
        encoder = keras.layers.Dense(i, activation='relu')(encoder)
    encoder = keras.layers.Dense(latent_dims, activation='relu')(encoder)

    decoder_input = keras.layers.Input(shape=latent_dims,)
    decoder = keras.layers.Dense(
        hidden_layers[-1], activation='relu'
    )(decoder_input)
    for i in (reversed(hidden_layers[:-1])):
        decoder = keras.layers.Dense(i, activation='relu')(decoder)
    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(decoder)

    encoder = keras.models.Model(input, encoder)
    decoder = keras.models.Model(decoder_input, decoder_output)
    output = decoder(encoder.output)
    auto = keras.models.Model(input, output)

    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
