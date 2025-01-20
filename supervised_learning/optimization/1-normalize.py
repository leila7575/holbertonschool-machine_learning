#!/usr/bin/env python3
"""This module contains the function normalize."""


import numpy as np


def normalize(X, m, s):
    """Normalizes a matrix."""
    X_norm = (X - np.mean(X, axis=0))/np.std(X, axis=0)
    return X_norm
