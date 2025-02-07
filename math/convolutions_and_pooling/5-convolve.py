#!/usr/bin/env python3
"""Contains convolves function."""


import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a valid, same or with custom padding
    convolution on images with multiple kernels."""
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h + 1) // 2
        pw = ((w - 1) * sw + kw - w + 1) // 2
    elif isinstance(padding, tuple):
        ph, pw = padding

    new_h = (h + 2 * ph - kh) // sh + 1
    new_w = (w + 2 * pw - kw) // sw + 1
    image_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant'
    )
    res = np.zeros((m, new_h, new_w, nc))

    for kernel in range(nc):
        for i in range(new_h):
            for j in range(new_w):
                res[:, i, j, kernel] = np.sum(
                    image_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                    * kernels[..., kernel],
                    axis=(1, 2, 3)
                )

    return res
