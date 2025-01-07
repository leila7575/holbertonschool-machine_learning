#!/usr/bin/env python3
"""This module defines a single neuron for binary classification."""

import numpy as np
import matplotlib.pyplot as plt


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
        """Evaluates the neuron's prediction.
        Returns the prediction and the cost."""
        A = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent."""
        m = Y.shape[1]
        dZ = A - Y
        dW = (1/m) * np.dot(dZ, X.T)
        db = (1/m) * np.sum(dZ)
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(
        self, X, Y, iterations=5000, alpha=0.05,
        verbose=True, graph=True, step=100
    ):
        """Trains the neuron over a certain number of iterations
        based on gradient descent and displays a graph
        representing the cost over iterations."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations < 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        elif alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            elif step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        y_values = []
        x_values = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            cost = self.cost(Y, A)
            self.gradient_descent(X, Y, A, alpha)
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
            array = np.arange(0, 3001, 500)
            plt.xticks(array)
            plt.show()

        return self.evaluate(X, Y)
