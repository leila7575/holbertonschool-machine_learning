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
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma.ravel() == 0] = 0
        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei
