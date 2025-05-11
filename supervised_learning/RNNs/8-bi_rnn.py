#!/usr/bin/env python3
"""Contains bi-rnn function for forward propagation"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward propagation for bidirectional RNN"""
    t, m, i = X.shape
    _, h = h_0.shape
    Hf = np.zeros([t + 1, m, h])
    Hf[0] = h_0

    Hb = np.zeros([t + 1, m, h])
    Hb[t] = h_t

    for i in range(t):
        H_next = bi_cell.forward(Hf[i], X[i])
        Hf[i + 1] = H_next

    for i in (reversed(range(t))):
        H_prev = bi_cell.backward(Hb[i + 1], X[i])
        Hb[i] = H_prev

    H = np.concatenate((Hf[1:], Hb[:t]), axis=2)
    Y = bi_cell.output(H)

    return H, Y
