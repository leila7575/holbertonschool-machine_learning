#!/usr/bin/env python3
"""Contains convolve_channels function."""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a valid, same or with custom padding
    convolution on images with channels."""
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2)
        pw = int(((w - 1) * sw + kw - w) / 2)
    elif isinstance(padding, tuple):
        ph, pw = padding

    new_h = int((h + 2 * ph - kh) / sh) + 1
    new_w = int((w + 2 * pw - kw) / sw) + 1
    image_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant'
    )
    res = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            res[:, i, j] = np.sum(
                image_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw] * kernel,
                axis=(1, 2, 3)
            )

    return res
