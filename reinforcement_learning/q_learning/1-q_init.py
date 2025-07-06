#!/usr/bin/env python3
"""Initializes Qtable with observation space(number of rows)
and action space(number of columns)"""


import numpy as np


def q_init(env):
    """Initializes Qtable with observation space and action space"""
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    return Q
