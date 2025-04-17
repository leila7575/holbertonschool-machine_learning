#!/usr/bin/env python3
"""contains class GaussianProcess"""


import numpy as np


class GaussianProcess:
    """Represents 1D Gaussian Process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """initializes Gaussian Process"""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """computes covariance kernel matrix based on RBF kernel"""
        squared_distance = (
            np.sum(X1**2, 1).reshape(-1, 1) + (
                np.sum(X2**2, 1)
            ) - 2 * np.dot(X1, X2.T)
        )
        K = self.sigma_f**2 * np.exp(- 0.5 / self.l**2 * squared_distance)
        return K

    def predict(self, X_s):
        """Predicts mean and standard deviation of new points
        based on gaussian process."""
        k = self.kernel(self.X, self.X)
        k_s = self.kernel(X_s, self.X)
        k_ss = self.kernel(X_s, X_s)
        mu = k_s.dot(np.linalg.inv(k)).dot(self.Y).flatten()
        sigma = k_ss - k_s.dot(np.linalg.inv(k)).dot(k_s.T)
        sigma = np.diag(sigma)
        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates gaussian process with new sample point X_new
        and new function value Y_new."""
        X_new = X_new.reshape(1, 1)
        Y_new = Y_new.reshape(1, 1)
        self.X = np.concatenate((self.X, X_new), axis=0)
        self.Y = np.concatenate((self.Y, Y_new), axis=0)
        cov_s = self.kernel(X_new, self.X[:-1])
        cov_ss = self.kernel(X_new, X_new)
        new_col = np.concatenate((cov_s.T, cov_ss), axis=0)
        self.K = np.concatenate((self.K, cov_s), axis=0)
        self.K = np.concatenate((self.K, new_col), axis=1)
