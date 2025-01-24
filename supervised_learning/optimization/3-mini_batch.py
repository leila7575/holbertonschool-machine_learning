#!/usr/bin/env python3
"""This module contains the function create_mini_batches."""


import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Creates mini-batches for mini-batch gradient descent."""
    X, Y = shuffle_data(X, Y)
    mini_batches = []
    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
