#!/usr/bin/env python3
"""Contains pdf function for GMM."""


import numpy as np


def pdf(X, m, S):
    """computes pdf for GMM."""

    try:
        n, d = X.shape
        X_centered = X - m
        det = np.linalg.det(S)
        cov_mat_inverse = np.linalg.inv(S)
        norm_const = 1 / (((2 * np.pi) ** d) * det) ** 0.5
        exp_val = np.sum(
            (np.matmul(X_centered, cov_mat_inverse)) * X_centered, axis=1
        )
        pdf_value = norm_const * np.exp(-0.5 * exp_val)
        P = np.maximum(pdf_value, 1e-300)
        return P
    except Exception:
        return None
