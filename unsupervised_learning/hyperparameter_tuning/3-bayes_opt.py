#!/usr/bin/env python3
"""contains class BayesianOptimization"""


import numpy as np

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
