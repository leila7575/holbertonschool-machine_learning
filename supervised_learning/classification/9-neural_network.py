#!/usr/bin/env python3
"""This module defines a neural network with one hidden layer."""

import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer
    for binary classification."""
    def __init__(self, nx, nodes):
        """Class constructor. Initializes the neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.nx = nx
        self.nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__W2 = np.random.randn(1, nodes)
        self.__b1 = np.zeros((nodes, 1))
        self.__b2 = 0
        self.__A1 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Retrieves the weights vector for the hidden layer."""
        return self.__W1

    @property
    def W2(self):
        """Retrieves the weights vector for the output neuron."""
        return self.__W2

    @property
    def b1(self):
        """Retrieves the bias for the hidden layer."""
        return self.__b1

    @property
    def b2(self):
        """Retrieves the bias for the output neuron."""
        return self.__b2

    @property
    def A1(self):
        """Retrieves the activated output for the hidden layer."""
        return self.__A1

    @property
    def A2(self):
        """Retrieves the activated output for the output neuron."""
        return self.__A2
