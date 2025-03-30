#!/usr/bin/env python3
"""Performs PCA on a dataset"""


import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset"""
    covariance_matrix = np.cov(X, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_eigenvalues = eigenvalues[np.argsort(eigenvalues)[::-1]]
    sorted_eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    total_variance = np.sum(sorted_eigenvalues)
    cum_variance = np.cumsum(sorted_eigenvalues) / total_variance
    number_dimensions = np.argmax(cum_variance >= var) + 1
    W = sorted_eigenvectors[:, :number_dimensions]
    return W
