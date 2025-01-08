#!/usr/bin/env python3
"""This module defines the one_hot_decode function"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels."""

    if not isinstance(one_hot, np.ndarray):
        return None

    label_vector = []

    for i in one_hot.T:
        label_vector.append(np.argmax(i))

    return np.array(label_vector)
