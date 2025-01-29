#!/usr/bin/env python3
"""This module contains the function precision."""


import numpy as np


def precision(confusion):
    """Returns a numpy.ndarray containing the precision of each class."""
    true_classes = confusion.shape[0]
    precision = []
    for i in range(true_classes):
        true_positive = confusion[i][i]
        predicted_positive = np.sum(confusion, axis=0)
        precision.append(true_positive/predicted_positive[i])
    return np.array(precision)
