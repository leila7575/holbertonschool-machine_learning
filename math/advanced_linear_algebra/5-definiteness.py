#!/usr/bin/env python3
"""Contains the function definiteness,
determines if the matrix is positive definite, positive semi-definite,
negative definite or negative semi-definite."""


import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if (not isinstance(matrix, np.ndarray) or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]):
        return None

    if not (matrix == matrix.T).all():
        return None

    eigenvalues, eigenvector = np.linalg.eig(matrix)

    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
