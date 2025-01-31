#!/usr/bin/env python3
"""This module contains the function l2_reg_cost."""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost with L2 regularization."""
    weight_squared_sum = 0
    for i in range(1, L+1):
        weight_squared_sum += np.linalg.norm(weights[f'W{i}'])**2
    cost_l2 = (lambtha / (2 * m)) * weight_squared_sum
    total_cost = cost + cost_l2
    return total_cost
