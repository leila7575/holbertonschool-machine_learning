#!/usr/bin/env python3
"""Contains variational autoencoder function"""


import tensorflow.keras as keras

def sampling(inputs):
    z_mean, z_log_var = inputs
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(z_mean))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder"""
    input = keras.Input(shape=(input_dims,))
    encoder_layer = input
    for i in hidden_layers:
        encoder_layer = keras.layers.Dense(i, activation='relu')(encoder_layer)
    
    z_mean = keras.layers.Dense(latent_dims, activation=None)(encoder_layer)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(encoder_layer)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.models.Model(input, [z_mean, z_log_var, z])

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoder_layer = decoder_input
    for i in (reversed(hidden_layers)):
        decoder_layer = keras.layers.Dense(i, activation='relu')(decoder_layer)
    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid'
    )(decoder_layer)

    decoder = keras.models.Model(decoder_input, decoder_output)
    output = decoder(encoder(input)[2])
    auto = keras.models.Model(input, output)

    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
