#!/usr/bin/env python3
"""Contains forward for HMM forward algorithm computation."""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model"""
    T = len(Observation)
    N = Initial.shape[0]

    F = np.zeros((T, N))
    F[0, :] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            for i in range(N):
                F[t, j] += (
                    F[t - 1, i]
                    * Transition[i, j]
                    * Emission[j, Observation[t]]
                )
    P = np.sum(F[T - 1, :])

    return P, F
