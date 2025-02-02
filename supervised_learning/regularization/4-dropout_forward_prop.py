#!/usr/bin/env python3
"""This module contains the function dropout_forward_prop."""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Performs forward propagation with dropout."""
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        A_prev = cache[f'A{i - 1}']
        W = weights[f'W{i}']
        b = weights[f'b{i}']

        Z = np.dot(W, A_prev) + b

        if i == L:
            e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = e_Z / np.sum(e_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)

        cache[f'A{i}'] = A

        if i < L:
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache[f'D{i}'] = D.astype(int)

    return cache
