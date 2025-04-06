#!/usr/bin/env python3
"""Contains initialization function for GMM."""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initialization function for GMM."""
    n, d = X.shape
    pi = np.ones(k) / k
    m, _ = kmeans(X, k)
    s = np.zeros((k, d, d))
    for i in range(k):
        for j in range(d):
            s[i, j, j] = 1
    return pi, m, s
