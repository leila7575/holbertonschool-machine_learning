#!/usr/bin/env python3
"""This module contains the function save_config and load_config."""


import tensorflow.keras as K


def save_config(network, filename):
    """saves the model's configuration in JSON format."""
    configuration = network.to_json()
    with open(filename, 'w') as f:
        f.write(configuration)
    return None


def load_config(filename):
    """loads a model with a determined configuration."""
    with open(filename, 'r') as f:
        configuration = f.read()
    model = K.models.model_from_json(configuration)
    return model
