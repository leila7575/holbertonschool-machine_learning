#!/usr/bin/env python3


import numpy as np


def correlation(C):
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d, d = C.shape
    variances = np.diag(C)
    standard_deviations = np.sqrt(variances)
    correlation_matrix = C / (
        standard_deviations[:, None] * standard_deviations[None, :]
    )
    return correlation_matrix
