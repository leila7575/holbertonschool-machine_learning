#!/usr/bin/env python3
"""Contains absorbing method to check if markov chain is absorbing."""


import numpy as np


def absorbing(P):
    """Determines if markov chain is absorbing"""
    n = P.shape[0]
    absorbing_states = np.where(np.isclose(P.diagonal(), 1))[0]
    if len(absorbing_states) == 0:
        return False

    trans_mat_power = np.linalg.matrix_power(P, n)

    for i in range(n):
        if i not in absorbing_states:
            if not np.any(trans_mat_power[i, absorbing_states] > 0):
                return False

    return True
