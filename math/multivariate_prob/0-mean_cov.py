#!/usr/bin/env python3
"""Contains the function mean_cov,
for mean and covariance computing."""


import numpy as np


def mean_cov(X):
    """Computes mean and covariance of a dataset."""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n, d = X.shape
    mean = np.mean(X, axis=0)
    mean = mean.reshape(1, -1)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
