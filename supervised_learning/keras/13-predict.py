#!/usr/bin/env python3
"""This module contains the function predict."""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Tests a model"""
    predictions = network.predict(data, verbose=verbose)
    return predictions
