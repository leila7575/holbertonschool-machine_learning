#!/usr/bin/env python3
"""Contains regular to determine stationary matrix"""


import numpy as np


def regular(P):
    """Computes steady state stationary matrix for regular markov chains."""
    try:
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        mask = ~np.isclose(eigenvalues, 1)
        if (
            np.any(np.isclose(eigenvalues, 1))
            and np.all(np.abs(eigenvalues[mask]) < 1)
            and np.sum(np.isclose(eigenvalues, 1)) == 1
        ):
            eigenvalues, eigenvectors = np.linalg.eig(P.T)
            eigenvectors_indexes = np.isclose(eigenvalues, 1)
            selected_eigenvect = eigenvectors[:, eigenvectors_indexes]
            selected_eigenvect = selected_eigenvect[:, 0]
            stationary_matrix = selected_eigenvect / np.sum(selected_eigenvect)
            return stationary_matrix
        return None
    except Exception:
        return None
