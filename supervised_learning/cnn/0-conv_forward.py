#!/usr/bin/env python3
"""Contains conv_forward, which performs convolutional forward propagation ."""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """Performs forward propagation on convolutional neural networks."""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev + 1) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev + 1) // 2

    new_h = (h_prev + 2 * ph - kh) // sh + 1
    new_w = (w_prev + 2 * pw - kw) // sw + 1
    image_padded = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant'
    )
    Z = np.zeros((m, new_h, new_w, c_new))

    for channel in range(c_new):
        for i in range(new_h):
            for j in range(new_w):
                Z[:, i, j, channel] = np.sum(
                    image_padded[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                    * W[..., channel],
                    axis=(1, 2, 3)
                ) + b[..., channel]
    activated_output = activation(Z)

    return activated_output
