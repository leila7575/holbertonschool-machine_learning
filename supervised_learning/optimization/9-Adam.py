#!/usr/bin/env python3
"""This module contains the function update_variables_Adam"""


import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable with Adam optimization algorithm."""
    first_moment = beta1 * v + (1 - beta1) * grad
    second_moment = beta2 * s + (1 - beta2) * (np.square(grad))
    vt_unbiased = first_moment / (1 - beta1**t)
    st_unbiased = second_moment / (1 - beta2**t)
    updated_var = var - alpha * vt_unbiased / (np.sqrt(st_unbiased) + epsilon)
    return updated_var, first_moment, second_moment
