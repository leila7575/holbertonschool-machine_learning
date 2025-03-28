#!/usr/bin/env python3
"""This module defines a neural network with one hidden layer."""

import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """Calculates forward propagation of neural network."""
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1/(1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1/(1 + np.exp(-Z2))
        return self.__A1, self.__A2

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
        _, A2 = self.forward_prop(X)
        prediction = (A2 >= 0.5).astype(int)
        cost = self.cost(Y, A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent."""
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neural network over a certain number of iterations
        based on gradient descent."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha < 0:
            raise ValueError("alpha must be positive")
        
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            elif step < 0 and step <= iterations:
                raise ValueError("step must be positive and <= iterations")

        y_values = []
        x_values = []
        
        for i in range(0, iterations + 1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)
            self.gradient_descent(X, Y, A1, A2, alpha)
            if verbose and i % step == 0:
                print(f"Cost after {i} iterations: {cost}")
            if graph and i % step == 0:
                y_values.append(cost)
                x_values.append(i)

        if graph:
            plt.plot(x_values, y_values)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            array = np.arange(0, 5001, 1000)
            plt.xticks(array)
            plt.show()

        return self.evaluate(X, Y)
