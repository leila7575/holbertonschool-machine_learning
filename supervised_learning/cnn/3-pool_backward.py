#!/usr/bin/env python3
"""Contains pool_backward, performs backward propagation on pooling layers."""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs backpropagation on pooling layers."""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros((A_prev.shape))

    for sample in range(m):
        for channel in range(c):
            for i in range(h_new):
                for j in range(w_new):
                    if mode == 'max':
                        A_prev_sliced = A_prev[
                            sample,
                            i*sh:i*sh+kh,
                            j*sw:j*sw+kw,
                            channel
                        ]
                        max_pooling = np.max(
                            A_prev_sliced == np.max(A_prev_sliced)
                        )
                        dA_prev[
                            sample,
                            i*sh:i*sh+kh,
                            j*sw:j*sw+kw,
                            channel
                        ] += np.multiply(
                            max_pooling,
                            dA[sample, i, j, channel]
                        )
                    elif mode == 'avg':
                        avg_poolig = np.ones(kernel_shape) * (dA[
                            sample,
                            i,
                            j,
                            channel
                        ]/(kh * kw))
                        dA_prev[
                            sample,
                            i*sh:i*sh+kh,
                            j*sw:j*sw+kw,
                            channel
                        ] += avg_poolig

    return dA_prev
