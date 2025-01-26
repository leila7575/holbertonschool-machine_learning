#!/usr/bin/env python3
"""This module contains the function create_batch_norm_layer."""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network."""
    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(mode='fan_avg'),
        )(prev)
    normalized_batch = tf.keras.layers.BatchNormalization(epsilon=1e-7)(layer)
    output = activation(normalized_batch)
    return output
