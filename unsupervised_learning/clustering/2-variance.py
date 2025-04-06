#!/usr/bin/env python3
"""Contains variance function."""


import numpy as np


def variance(X, C):
    """Computes total intra-cluster variance for a dataste."""
    try:
        n, d = X.shape
        k, _ = C.shape
        euclidean_dist = np.sum(
            ((X[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2), axis=2
        )
        closest_centroid = np.argmin(euclidean_dist, axis=1)
        centroids = C[closest_centroid]
        var = np.sum(np.sum((X - centroids) ** 2, axis=1))
        return var
    except Exception:
        return None
