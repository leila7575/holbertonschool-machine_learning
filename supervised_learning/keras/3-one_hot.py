#!/usr/bin/env python3
"""This module contains the function optimize_model."""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    if classes is None:
        classes = max(labels) + 1
    one_hot_matrix = K.utils.to_categorical(labels, classes)
    return one_hot_matrix
