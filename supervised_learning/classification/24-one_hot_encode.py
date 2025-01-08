#!/usr/bin/env python3
"""This module defines the one_hot_encode function"""

import numpy as np


def one_hot_encode(Y, classes):
    """Converts a numeric label vector into a one-hot matrix."""
    m = Y.shape[0]
    matrix = np.zeros((classes, m))
    
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None

    for i in range(m):
        matrix[Y[i], i] = 1

    return matrix
