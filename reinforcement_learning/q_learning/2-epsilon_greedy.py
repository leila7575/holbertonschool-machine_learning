#!/usr/bin/env python3
"""Chooses next action based on epsilon-greedy policy."""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Chooses next action based on epsilon-greedy policy."""
    p = np.random.uniform()
    number_actions = Q.shape[1]
    if p < epsilon:
        return np.random.randint(number_actions)
    else:
        return np.argmax(Q[state][:])
