#!/usr/bin/env python3
"""Contains function intersection."""


import numpy as np


def factorial(n):
    """Computes factorial of n."""
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


def intersection(x, n, P, Pr):
    """Calculates intersection of obtaining x and n for each probability p."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if any(value < 0 for value in P) or any(value > 1 for value in P):
        raise ValueError(f"All values in P must be in the range [0, 1]")

    if any(value < 0 for value in Pr) or any(value > 1 for value in Pr):
        raise ValueError(f"All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1.0):
        raise ValueError("Pr must sum to 1")

    likelihood = []
    intersection = []
    binomial_coefficient = factorial(n) / (factorial(x) * factorial(n - x))
    for i in range(len(P)):
        PMF = binomial_coefficient * (P[i] ** x) * ((1 - P[i]) ** (n - x))
        likelihood.append(PMF)
        intersection.append(likelihood[i] * Pr[i])

    return np.array(intersection)
