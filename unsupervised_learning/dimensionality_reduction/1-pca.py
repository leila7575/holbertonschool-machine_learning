#!/usr/bin/env python3
"""Performs PCA on a dataset"""


import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset with SVD"""
    X_centered = X - np.mean(X, axis=0)
    left_sing_vect, sing_values, right_sing_vect = np.linalg.svd(X_centered)
    W = right_sing_vect[:ndim].T
    T = np.matmul(X_centered, W)

    return T
