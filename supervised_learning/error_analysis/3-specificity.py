#!/usr/bin/env python3
"""This module contains the function specificity."""


import numpy as np


def specificity(confusion):
    """Returns a numpy.ndarray containing the specificity of each class."""
    true_classes = confusion.shape[0]
    specificity = []
    for i in range(true_classes):
        true_positive = confusion[i][i]
        positive = np.sum(confusion, axis=1)
        predicted_positive = np.sum(confusion, axis=0)
        total_sum = np.sum(confusion, axis=None)
        true_negative = total_sum - (
            positive[i] + predicted_positive[i] - true_positive
            )
        specificity.append(true_negative/(total_sum - positive[i]))
    return np.array(specificity)
