#!/usr/bin/env python3
"""This module contains function dropout_create_layer
which creates a layer including dropout."""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Creates a layer with dropout."""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg'
        )
    )(prev)

    if training:
        layer = tf.keras.layers.Dropout(rate=1 - keep_prob)(
            layer, training=True
        )
    return layer
