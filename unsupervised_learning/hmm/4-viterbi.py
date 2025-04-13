#!/usr/bin/env python3
"""Contains viterbi for HMM viterbi algorithm computation."""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """performs the viterbi algorithm for computing sequence of hidden states
    in a hidden markov model"""
    try:
        T = len(Observation)
        N = Initial.shape[0]

        V = np.zeros((N, T))
        best_states = np.zeros((N, T))
        V[:, 0] = Emission[:, Observation[0]] * Initial[:, 0]
        for t in range(1, T):
            for i in range(N):
                Vt = V[:, t - 1] * Transition[:, i]
                best_state = np.argmax(Vt)
                V[i, t] = Vt[best_state] * Emission[i, Observation[t]]
                best_states[i, t] = best_state

        end_state = np.argmax(V[:, T - 1])
        path = np.zeros(T, dtype=int)
        path[T - 1] = end_state
        P = V[end_state, T - 1]

        for t in range(T - 2, -1, -1):
            path[t] = best_states[path[t + 1], t + 1]

        return path, P
    except Exception:
        return None, None
