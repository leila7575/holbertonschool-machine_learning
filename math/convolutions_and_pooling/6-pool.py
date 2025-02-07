#!/usr/bin/env python3
"""Contains pool function, which performs pooling on images."""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs max or average pooling on images."""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = (h - kh) // sh + 1
    new_w = (w - kw) // sw + 1

    res = np.zeros((m, new_h, new_w, c))

    for i in range(new_h):
        for j in range(new_w):
            if mode == 'max':
                res[:, i, j, :] = np.max(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :],
                    axis=(1, 2)
                    )
            if mode == 'avg':
                res[:, i, j, :] = np.average(
                    images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :],
                    axis=(1, 2)
                    )

    return res
