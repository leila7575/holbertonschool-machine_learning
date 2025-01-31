#!/usr/bin/env python3
"""This module contains function l2_reg_create_layer with creates a layer including L2 regularization."""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a layer with L2 regularization."""
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg'),
        kernel_regularizer=tf.keras.regularizers.L2(l=lambtha)
        )(prev)
    return layer