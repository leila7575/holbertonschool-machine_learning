#!/usr/bin/env python3
"""This module defines the one_hot_encode function"""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix."""
    if not isinstance(classes, int):
        return None

    Y = np.array(Y)
    m = Y.shape[0]
    matrix = np.zeros((classes, m))

    for i in range(m):
        matrix[Y[i], i] = 1

    return matrix
