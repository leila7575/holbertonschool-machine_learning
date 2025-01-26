#!/usr/bin/env python3
"""This module contains the function nobatch_norm."""


import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes matrix Z using batch normalization."""
    Z_norm = (Z - np.mean(Z, axis=0))/np.sqrt((np.var(Z, axis=0) + epsilon))
    Z_final = gamma * Z_norm + beta
    return Z_final
