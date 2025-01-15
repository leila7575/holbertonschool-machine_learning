#!/usr/bin/env python3
"""This module contains the function forward_prop."""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network."""
    input_data = x

    for i, layer_size in enumerate(layer_sizes):
        activation = activations[i]
        input_data = create_layer(input_data, layer_size, activation)
    return input_data
