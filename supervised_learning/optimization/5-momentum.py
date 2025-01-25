#!/usr/bin/env python3
"""This module contains the function update_variables_momentum."""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """Updates a variable with gradient descent with momentum algorithm."""
    velocity = beta1 * v + (1 - beta1) * grad
    updated_variable = var - alpha * velocity
    return updated_variable, velocity
