#!/usr/bin/env python3
"""This module contains the function dropout_gradient_descent."""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights with gradient descent and dropout regularization."""
    m = Y.shape[1]
    gradients = {}

    A = cache[f'A{L}']
    A_prev = cache[f'A{L - 1}']
    dZ = A - Y

    gradients[f'dW{L}'] = (1 / m) * np.dot(dZ, A_prev.T)
    gradients[f'db{L}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    for i in range(L - 1, 0, -1):
        A = cache[f'A{i}']
        A_prev = cache[f'A{i - 1}']
        dZ_curr = np.multiply(np.dot(weights[f'W{i + 1}'].T, dZ), 1 - A**2)

        gradients[f'dW{i}'] = (1 / m) * np.dot(dZ_curr, A_prev.T)
        gradients[f'db{i}'] = (1 / m) * np.sum(dZ_curr, axis=1, keepdims=True)

        D = cache[f'D{i}']
        dZ_curr = dZ_curr * D
        dZ_curr /= keep_prob

        dZ = dZ_curr

    for i in range(1, L + 1):
        weights[f'W{i}'] -= alpha * gradients[f'dW{i}']
        weights[f'b{i}'] -= alpha * gradients[f'db{i}']
