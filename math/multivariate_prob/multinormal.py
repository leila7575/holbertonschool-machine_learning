#!/usr/bin/env python3
"""Contains class Multinormal."""


import numpy as np


class MultiNormal:
    """Represents a multivariate normal distributionn."""
    def __init__(self, data):
        """Class constructor for multinormal distribution class."""
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        self.cov = np.matmul(
            (data - self.mean), (data - self.mean).T
        ) / (n - 1)

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d, _ = x.shape

        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        det = np.linalg.det(self.cov)
        cov_mat_inverse = np.linalg.inv(self.cov)
        norm_const = 1 / (((2 * np.pi) ** d) * det) ** 0.5
        pdf_value = norm_const * (
            np.exp(-0.5 * (np.matmul(
                np.matmul((x - self.mean).T, cov_mat_inverse), (x - self.mean)
            )))
        )

        return float(pdf_value)
