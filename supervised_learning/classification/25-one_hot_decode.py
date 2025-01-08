#!/usr/bin/env python3
"""This module defines the one_hot_decode function"""

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels."""

    if not isinstance(one_hot, np.ndarray):
        return None

    label_vector = np.argmax(one_hot, axis=0)

    return label_vector
