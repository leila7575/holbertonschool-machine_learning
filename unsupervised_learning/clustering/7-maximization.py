#!/usr/bin/env python3
"""Contains maximization function for GMM."""


import numpy as np


def maximization(X, g):
    """computes maximization step for GMM."""
    n, d = X.shape
    k = g.shape[0]
    pi = np.sum(g, axis=1) / n
    m = np.dot(g, X) / np.sum(g, axis=1)[:, None]

    S = np.zeros((k, d, d))
    for i in range(k):
        X_centered = X - m[i]
        S[i] = np.dot(g[i] * X_centered.T, X_centered) / np.sum(g[i])

    return pi, m, S
