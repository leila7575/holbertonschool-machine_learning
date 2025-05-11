#!/usr/bin/env python3
"""Class LSTMCell"""


import numpy as np


class LSTMCell:
    """LSTM unit"""
    def __init__(self, i, h, o):
        """class constructor, initializes LSTM Cell attributes"""
        self.i = i
        self.h = h
        self.o = o
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros(shape=(1, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros(shape=(1, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros(shape=(1, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros(shape=(1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros(shape=(1, o))

    def sigmoid(self, x):
        """Applies sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """Applies softmax function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one step"""
        concatenated_h = np.concatenate((h_prev, x_t), axis=1)
        ft = self.sigmoid(np.matmul(concatenated_h, self.Wf) + self.bf)
        ut = self.sigmoid(np.matmul(concatenated_h, self.Wu) + self.bu)
        c_bar = np.tanh(np.matmul(concatenated_h, self.Wc) + self.bc)
        c_next = ft * c_prev + ut * c_bar
        ot = self.sigmoid(np.matmul(concatenated_h, self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, c_next, y
