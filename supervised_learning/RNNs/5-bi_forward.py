#!/usr/bin/env python3
"""Class BidirectionalCell"""


import numpy as np


class BidirectionalCell:
    """Bidirectional cell in RNN"""
    def __init__(self, i, h, o):
        """class constructor, initializes Bidirectional Cell attributes"""
        self.i = i
        self.h = h
        self.o = o
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros(shape=(1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one step"""
        concatenated_h = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concatenated_h, self.Whf) + self.bhf)
        return h_next
