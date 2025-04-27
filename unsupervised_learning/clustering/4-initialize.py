#!/usr/bin/env python3
"""Contains initialization function for GMM."""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initialization function for GMM."""
    try:
        n, d = X.shape
        pi = np.ones(k) / k
        m, _ = kmeans(X, k)
        s = np.eye(d)
        s = np.tile(s, (k, 1, 1))
        return pi, m, s
    except Exc
