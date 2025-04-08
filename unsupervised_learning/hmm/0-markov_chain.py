#!/usr/bin/env python3
"""Contains markov_chain to determine probability of markov chain."""


import numpy as np


def markov_chain(P, s, t=1):
    """Computes state distribution matrix of a markov chain
    after t iterations."""
    try:
        state_distribution_mat = np.matmul(s, np.linalg.matrix_power(P, t))
        return state_distribution_mat
    except Exception:
        return None
