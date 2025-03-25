#!/usr/bin/env python3
"""Contains function likelihood."""


import numpy as np


def factorial(n):
    """Computes factorial of n."""
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def likelihood(x, n, P):
    """Calculates likelihood of obtaining x and n for each probability p."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("p must be a 1D numpy.ndarray")

    if any(value < 0 for value in P) or any(value > 1 for value in P):
        raise ValueError("All values in P must be in the range[0, 1]")

    likelihood = []
    binomial_coefficient = factorial(n) / (factorial(x) * factorial(n - x))
    for i in range(len(P)):
        PMF = binomial_coefficient * (P[i] ** x) * ((1 - P[i]) ** (n - x))
        likelihood.append(PMF)
    return likelihood
