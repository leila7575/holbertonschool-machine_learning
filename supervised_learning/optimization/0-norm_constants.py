#!/usr/bin/env python3
"""This module contains the function normalization_constants."""


import numpy as np


def normalization_constants(X):
    """Calculates normalization constants
    Returns mean and standard deviation for each feature."""
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    return m, s
