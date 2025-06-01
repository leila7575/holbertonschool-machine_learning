#!/usr/bin/env python3
"""Scaled dot product attention"""


import numpy as np
import tensorflow as tf


def softmax(x):
    """computes softmax activation function."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sdp_attention(Q, K, V, mask=None):
    """"Computes scaled dot product attention."""
    scores = np.matmul(
        Q, np.transpose(K, axes=(*range(K.ndim-2), -1, -2))
    ) / np.sqrt(K.shape[-1])

    if mask is not None:
        scores += -1e9 * mask

    weights = softmax(scores)
    output = np.matmul(weights, V)
    return output, weights
