#!/usr/bin/env python3
"""Contains convolve_grayscale_padding function."""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on a grayscale image with custom padding."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    res = np.zeros((m, h + 2 * ph - kh + 1, w + 2 * pw - kw + 1))
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    for i in range(h + 2 * ph - kh + 1):
        for j in range(w + 2 * pw - kw + 1):
            res[:, i, j] = np.sum(
                image_padded[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )

    return res
