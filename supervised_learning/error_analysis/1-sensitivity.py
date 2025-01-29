#!/usr/bin/env python3
"""This module contains the function sensitivity."""


import numpy as np


def sensitivity(confusion):
    """Returns a numpy.ndarray containing the sensitivity of each class."""
    true_classes = confusion.shape[0]
    sensitivity = []
    for i in range(true_classes):
        true_positive = confusion[i][i]
        positive = np.sum(confusion, axis=1)
        sensitivity.append(true_positive/positive[i])
    return sensitivity
