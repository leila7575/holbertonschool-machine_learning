#!/usr/bin/env python3
"""Contains baum-welch for HMM baum-welch algorithm computation."""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """Performs the forward algorithm for a hidden Markov model."""
    T = len(Observation)
    N = Initial.shape[0]

    F = np.zeros((N, T))
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(
                F[:, t - 1] * Transition[:, j]
            ) * Emission[j, Observation[t]]
    return F


def backward(Observation, Emission, Transition, Initial):
    """Performs the backward algorithm for a hidden Markov model."""
    T = len(Observation)
    N = Initial.shape[0]

    B = np.zeros((N, T))
    B[:, T - 1] = 1
    for t in reversed(range(T - 1)):
        for i in range(N):
            B[i, t] = np.sum(
                Transition[i, :] *
                Emission[:, Observation[t + 1]] *
                B[:, t + 1]
            )
    return B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """Performs the Baum-Welch algorithm for a hidden Markov model."""
    try:
        T = Observations.shape[0]
        M, N = Emission.shape

        for _ in range(iterations):
            alpha = forward(Observations, Emission, Transition, Initial)
            beta = backward(Observations, Emission, Transition, Initial)

            xi = np.zeros((M, M, T - 1))
            for t in range(T - 1):
                denominator = np.sum(
                    alpha[:, t] *
                    (Transition @ (
                        Emission[:, Observations[t + 1]] * beta[:, t + 1]
                    ))
                )

                for i in range(M):
                    numerator = (
                        alpha[i, t]
                        * Transition[i, :]
                        * Emission[:, Observations[t + 1]]
                        * beta[:, t + 1]
                    )
                    xi[i, :, t] = numerator / denominator

            gamma = np.sum(xi, axis=1)
            final_gamma = (alpha[:, T - 1] * beta[:, T - 1]) / (np.sum(alpha[:, T - 1] * beta[:, T - 1]) + 1e-8)
            gamma = np.hstack((gamma, final_gamma[:, None]))
            Initial = gamma[:, [0]]

            Transition = np.sum(xi, axis=2) / np.sum(
                gamma[:, :-1], axis=1, keepdims=True
            )

            for k in range(N):
                mask = Observations == k
                Emission[:, k] = np.sum(gamma[:, mask], axis=1)
            Emission = Emission / np.sum(gamma, axis=1, keepdims=True)

        return Transition, Emission

    except Exception:
        return None, None
