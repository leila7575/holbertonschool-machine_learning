#!/usr/bin/env python3
"""Contains conv_backward, performs convolutional backward propagation."""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Performs backpropagation on convolutional neural networks."""
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int(np.floor(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.floor(((w_prev - 1) * sw + kw - w_prev) / 2))

    A_prev_padded = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant'
    )
    dA_prev = np.zeros((A_prev.shape))
    dA_prev_padded = np.pad(
        dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant'
    )
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for sample in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for channel in range(c_new):
                    dA_prev_padded[sample, i*sh:i*sh+kh, j*sw:j*sw+kw, :] += (
                        W[..., channel] * dZ[sample, i, j, channel]
                    )
                    dW[..., channel] += (
                        A_prev_padded[sample, i*sh:i*sh+kh, j*sw:j*sw+kw, :] *
                        dZ[sample, i, j, channel]
                    )

    if padding == 'same':
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
