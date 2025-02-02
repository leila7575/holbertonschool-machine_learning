#!/usr/bin/env python3
"""This module contains the function dropout_gradient_descent."""


import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights with gradient descent and dropout regularization."""
    m = Y.shape[1]
    grad = {}

    dZ = cache[f'A{L}'] - Y
    for i in reversed(range(1, L + 1)):
        A_prev = cache[f'A{i - 1}']
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        if i < L:
            dW += (lambtha / m) * W
            
        grad[f'dW{i}'] = dW
        grad[f'db{i}'] = db
        
        if i > 1:
            D = cache[f'D{i}']
            dA_prev = np.dot(W.T, dZ)
            dA_prev *= D
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - np.square(A_prev))

        weights[f'W{i}'] -= alpha * grad[f'dW{i}']
        weights[f'b{i}'] -= alpha * grad[f'db{i}']

    return weights