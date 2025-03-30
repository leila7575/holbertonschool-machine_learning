#!/usr/bin/env python3
"""Performs PCA on a dataset"""


import numpy as np


def pca(X, ndim):
    """Performs PCA on a dataset with SVD"""
    left_sing_vect, sing_values, right_sing_vect = np.linalg.svd(X)
    W = right_sing_vect.T[:, :ndim]
    T = np.matmul(X, W)

    return T
