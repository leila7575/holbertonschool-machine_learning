#!/usr/bin/env python3
"""Contains deep_rnn function for forward propagation"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep RNN"""
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    o = rnn_cells[-1].by.shape[1]
    Y = np.zeros((t, m, o))
    for i in range(t):
        input = X[i]
        for j in range(l):
            H_next, Y_t = rnn_cells[j].forward(H[i, j], input)
            H[i + 1, j] = H_next
            input = H_next
        Y[i] = Y_t
    return H, Y
