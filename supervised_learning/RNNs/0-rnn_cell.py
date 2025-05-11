#!/usr/bin/env python3
"""Class RNNCell"""


import numpy as np


class RNNCell:
    """Simple RNN cell"""
    def __init__(self, i, h, o):
        """class constructor, initializes RNNCell attributes"""
        self.i = i
        self.h = h
        self.o = o
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def softmax(self, x):
        """Applies softmax function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one step"""
        concatenated_h = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(concatenated_h, self.Wh) + self.bh)
        x = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(x)
        return h_next, y
