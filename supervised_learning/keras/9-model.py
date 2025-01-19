#!/usr/bin/env python3
"""This module contains the function save_model."""


import tensorflow.keras as K


def save_model(network, filename):
    """saves the model."""
    network.save(filename)


def load_model(filename):
    """loads the model."""
    model = K.models.load_model(filename)
    return model
