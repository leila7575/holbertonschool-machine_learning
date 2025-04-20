#!/usr/bin/env python3
"""contains class BayesianOptimization"""


import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on 1D Gaussian Process"""
    def __init__(
        self, f, X_init, Y_init, bounds, ac_samples, l=1,
        sigma_f=1, xsi=0.01, minimize=True
    ):
        """Class constructor for Bayesian optimization"""
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.f = f
        self.bounds = bounds
        self.ac_samples = ac_samples
        self.l = l
        self.sigma_f = sigma_f
        self.xsi = xsi
        self.minimize = minimize
        step = (bounds[1] - bounds[0]) / (ac_samples - 1)
        X_s = np.arange(
            bounds[0], bounds[1] + step / 2, step
        ).reshape(ac_samples, 1)
        self.X_s = X_s

    def acquisition(self):
        """Computes next best sample point
        with Expected Improvement function."""
        mu, sigma = self.gp.predict(self.X_s)
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        if self.minimize:
            best_pred = np.min(self.gp.Y)
            improvement = best_pred - mu - self.xsi
        else:
            best_pred = np.max(self.gp.Y)
            improvement = mu - best_pred - self.xsi
        with np.errstate(divide='warn'):
            Z = np.zeros_like(improvement)
            non_zero = sigma > 0
            Z[non_zero] = improvement[non_zero] / sigma[non_zero]
            ei = np.zeros_like(mu)
            cdf = norm.cdf(Z[non_zero])
            pdf = norm.pdf(Z[non_zero])
            ei[non_zero] = improvement[non_zero] * cdf + sigma[non_zero] * pdf
            ei[~non_zero] = 0
        X_next = self.X_s[np.argmax(ei)].reshape(-1, 1)
        return X_next, ei

    def optimize(self, iterations=100):
        """Optimizes the function f using Bayesian optimization"""
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if any(np.allclose(X_next, x) for x in self.gp.X):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
        if self.minimize:
            best_index = np.argmin(self.gp.Y)
        else:
            best_index = np.argmax(self.gp.Y)
        X_opt = self.gp.X[best_index]
        Y_opt = self.gp.Y[best_index]
        return X_opt, Y_opt
