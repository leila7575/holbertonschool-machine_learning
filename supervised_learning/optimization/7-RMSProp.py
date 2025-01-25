#!/usr/bin/env python3
"""This module contains the function update_variables_RMSProp."""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """Updates a variable with RMSProp algorithm."""
    second_moment = beta2 * s + (1 - beta2) * (np.square(grad))
    updated_variable = var - alpha * grad / (np.sqrt(second_moment) + epsilon)
    return updated_variable, second_moment
