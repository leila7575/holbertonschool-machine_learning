#!/usr/bin/env python3
"""This module contains the function f1_score."""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Returns a numpy.ndarray containing the f1_score of each class
    based on sensitivity and precision."""
    true_classes = confusion.shape[0]
    fscore = []
    ppv = precision(confusion)
    recall = sensitivity(confusion)
    for i in range(true_classes):
        fscore.append(2 * ppv[i] * recall[i] / (ppv[i] + recall[i]))
    return np.array(fscore)
