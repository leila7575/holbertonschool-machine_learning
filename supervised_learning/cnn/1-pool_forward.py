#!/usr/bin/env python3
"""Contains function pool_forward"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs forward propagation on a pooling layer."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    new_h = (h_prev - kh) // sh + 1
    new_w = (w_prev - kw) // sw + 1

    res = np.zeros((m, new_h, new_w, c_prev))

    for i in range(new_h):
        for j in range(new_w):
            if mode == 'max':
                res[:, i, j, :] = np.max(
                    A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :],
                    axis=(1, 2)
                    )
            if mode == 'avg':
                res[:, i, j, :] = np.average(
                    A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :],
                    axis=(1, 2)
                    )

    return res
