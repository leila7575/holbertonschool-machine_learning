#!/usr/bin/env python3
"""This module contains the function create_batch_norm_layer."""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network."""
    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
            )
        )(prev)
    mean, variance = tf.nn.moments(layer, axes=[0])
    gamma = tf.Variable(initial_value=tf.ones([n]), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros([n]), trainable=True)
    normalized_layer = (layer - mean) / tf.sqrt(variance + 1e-7)
    final_layer = gamma * normalized_layer + beta
    output = activation(final_layer)
    return output
