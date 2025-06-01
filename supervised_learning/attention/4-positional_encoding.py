#!/usr/bin/env python3
"""Positional encoding"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """Computes positional encoding for transformer"""
    P = np.zeros((max_seq_len, dm))
    for k in range(max_seq_len):
        for i in np.arange(int(dm/2)):
            denominator = np.power(10000, 2*i/dm)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return np.array(P)
