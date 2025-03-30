#!/usr/bin/env python3
"""Performs PCA on a dataset"""


import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset with SVD"""
    left_sing_vect, sing_values, right_sing_vect = np.linalg.svd(X)

    variance_ratios = sing_values / np.sum(sing_values)
    cum_variance = np.cumsum(variance_ratios)

    number_dimensions = np.searchsorted(cum_variance, var) + 1

    W = right_sing_vect.T[:, :number_dimensions]

    return W
