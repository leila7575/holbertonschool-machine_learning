#!/usr/bin/env python3
"""This module contains the function save_model."""


import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """saves the model's weights."""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads the model's weights."""
    model = K.models.load_weights(filename)
    return None
