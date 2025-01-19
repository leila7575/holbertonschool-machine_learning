#!/usr/bin/env python3
"""This module contains the function test_model."""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a model"""
    test_loss, test_accuracy = network.evaluate(data, labels, verbose=verbose)
    return test_loss, test_accuracy
