#!/usr/bin/env python3
"""Contains convolve_grayscale function."""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a valid, same or with custom padding
    convolution on a grayscale image."""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif isinstance(padding, tuple):
        ph, pw = padding

    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1
    image_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    res = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            res[:, i, j] = np.sum(
                image_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel,
                axis=(1, 2)
            )

    return res
