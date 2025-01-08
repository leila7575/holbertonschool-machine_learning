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

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        layer_prev = nx
        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f'W{i+1}'] = np.random.randn(
                layers[i], layer_prev
                ) * np.sqrt(2/layer_prev)
            self.weights[f'b{i+1}'] = np.zeros((layers[i], 1))
            layer_prev = layers[i]
