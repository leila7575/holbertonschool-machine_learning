#!/usr/bin/env python3
"""This module defines a single neuron for binary classification."""

import numpy as np


class Neuron:
    """Defines a single neuron for binary classification."""
    def __init__(self, nx):
        """Class constructor. Initializes the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be positive")
        self.nx = nx
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Retrieves the weights vector."""
        return self.__W

    @property
    def b(self):
        """Retrieves the bias."""
        return self.__b

    @property
    def A(self):
        """Retrieves the activated output."""
        return self.__A

    def forward_prop(self, X):
        """Calculates forward propagation of the neuron."""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1/(1 + np.exp(-Z))
        return self.__A
