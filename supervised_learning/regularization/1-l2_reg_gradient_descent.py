#!/usr/bin/env python3
"""This module contains the function l2_reg_gradient_descent."""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Weight and bias updates with L2 regularization."""
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y
    for i in reversed(range(1, L + 1)):
        A_prev = cache[f'A{i - 1}'] if i > 1 else cache['A0']
        W = weights[f'W{i}']
        b = weights[f'b{i}']

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - np.square(A_prev))
