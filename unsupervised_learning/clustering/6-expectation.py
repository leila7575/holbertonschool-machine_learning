#!/usr/bin/env python3
"""Contains expectation function for GMM."""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """computes expectation step for GMM."""
    try:
        n, d = X.shape
        k = pi.shape[0]
        g = np.zeros((k, n))
        for i in range(k):
            P = pdf(X, m[i], S[i])
            if np.any(np.isnan(P)):
                return None, None
            g[i] = P * pi[i]
        total_prob = np.sum(g, axis=0)

        g /= total_prob
        l = np.sum(np.log(total_prob))
        return g, l
    except Exception:
        return None, None
