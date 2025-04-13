#!/usr/bin/env python3
"""Contains backward for HMM backward algorithm computation."""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model"""
    T = len(Observation)
    N = Initial.shape[0]

    B = np.zeros((T, N))
    B[T - 1, :] = 1
    for t in reversed(range(0, T - 1)):
        for j in range(N):
            for i in range(N):
                B[t, i] += (
                    B[t + 1, j]
                    * Transition[i, j]
                    * Emission[j, Observation[t + 1]]
                )
    P = np.sum(B[0, :] * Initial[:, 0] * Emission[:, Observation[0]])

    return P, B.T
