#!/usr/bin/env python3
"""This module contains the function create_placeholder."""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """Creates layer and returns tensor output"""
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
            ),
        name='layer',
    )(prev)

    return layer
