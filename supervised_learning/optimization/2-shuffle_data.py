#!/usr/bin/env python3
"""This module contains the function shuffle."""


import numpy as np


def shuffle_data(X, Y):
    """shuffles data points in two matrices."""
    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return X[p], Y[p]
