#!/usr/bin/env python3
"""This module defines a deep neural network"""

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network for binary classification."""
    def __init__(self, nx, layers):
        """Class constructor. Initializes the neural network."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        elif any(not isinstance(ele, int) or ele < 0 for ele in layers):
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if i == 0:
                self.weights['W1'] = np.random.randn(
                    layers[i], nx
                    ) * np.sqrt(2/nx)
            else:
                self.weights[f'W{i+1}'] = np.random.randn(
                    layers[i], layers[i - 1]
                    ) * np.sqrt(2/(layers[i-1]))
            self.weights[f'b{i+1}'] = np.zeros((layers[i], 1))
