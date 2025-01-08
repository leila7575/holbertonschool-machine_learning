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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer_prev = nx
        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")
            self.weights[f'W{i+1}'] = np.random.randn(
                layers[i], layer_prev
                ) * np.sqrt(2/layer_prev)
            self.weights[f'b{i+1}'] = np.zeros((layers[i], 1))
            layer_prev = layers[i]

    @property
    def L(self):
        """Retrieves the number of layers of the neural network."""
        return self.__L

    @property
    def cache(self):
        """Retrieves the cache storing intermediate values."""
        return self.__cache

    @property
    def weights(self):
        """Retrieves the dictionary containing weights and bias."""
        return self.__weights

    def forward_prop(self, X):
        """Calculates forward propagation of the neural network."""
        self.__cache['A0'] = X
        for i in range(1, self.L + 1):
            Z = np.dot(
                self.__weights[f'W{i}'], self.__cache[f'A{i - 1}']
                ) + self.__weights[f'b{i}']
            A = 1/(1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates cost of the model based on logistic regression"""
        m = Y.shape[1]
        cost = -1/m * np.sum(
            np.multiply(np.log(A), Y) + np.multiply(
                np.log(1.0000001 - A), 1 - Y
                )
            )
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network's prediction.
        Returns the prediction and the cost."""
        A, _ = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost
