#!/usr/bin/env python3
"""Contains rnn function for forward propagation"""


import numpy as np
rnn_cell = __import__('0-rnn_cell').RNNCell


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN"""
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros([t + 1, m, h])
    H[0] = h_0
    _, Y_0 = rnn_cell.forward(h_0, X[0])
    o = Y_0.shape[1]
    Y = np.zeros([t, m, o])
    for i in range (t):
        H_next, Y_t = rnn_cell.forward(X[i,:,:], H[i])
        H[i + 1,:,:] = H_next
        Y[i,:,:] = Y_t
    return H, Y
