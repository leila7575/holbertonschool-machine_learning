#!/usr/bin/env python3
"""Contains convolve_grayscale_same function."""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a convolution applying a kernel on a grayscale image."""
    m, h, w = images.shape
    kh, kw = kernel.shape

    res = np.zeros((m, h, w))
    image_padded = np.zeros((m, h + 2, w + 2))
    image_padded[:, 1:-1, 1:-1] = images
    for i in range(h):
        for j in range(w):
            res[:, i, j] = np.sum(image_padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return res