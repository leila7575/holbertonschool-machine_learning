#!/usr/bin/env python3
"""Contains convolve_grayscale_valid function."""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a convolution applying a kernel on a grayscale image."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    res = np.zeros((m, h - kh + 1, w - kw + 1))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            res[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return res
