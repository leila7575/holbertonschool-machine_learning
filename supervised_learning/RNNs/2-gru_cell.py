#!/usr/bin/env python3
"""Class GRUCell"""


import numpy as np


class GRUCell:
    """Gated recurrent unit"""
    def __init__(self, i, h, o):
        """class constructor, initializes GRUCell attributes"""
        self.i = i
        self.h = h
        self.o = o
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros(shape=(1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros(shape=(1, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one step"""
        concatenated_h = np.concatenate((h_prev, x_t), axis=1)
        zt = self.sigmoid(np.dot(concatenated_h, self.Wz) + self.bz)
        rt = self.sigmoid(np.dot(concatenated_h, self.Wr) + self.br)
        concatenated_rt = np.concatenate((x_t, (rt * h_prev)), axis=1)
        h_bar = np.tanh(np.dot(concatenated_rt, self.Wh) + self.bh)
        h_next = (np.ones_like(zt) - zt) * h_bar + zt * h_prev
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, y
