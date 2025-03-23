#!/usr/bin/env python3
"""Contains class Binomial."""


class Binomial:
    """Represents a binomial distribution."""
    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor for normal distribution class."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)

            variance = (sum((x - mean) ** 2 for x in data)) / len(data)
            q = variance / mean
            p = (1 - q)
            n = round(mean / p)
            p = float(mean / n)
            self.n = n
            self.p = p

    @staticmethod
    def factorial(n):
        """Calculates factorial of n"""
        factorial = 1
        for exp in range(2, n + 1):
            factorial *= exp
        return factorial

    def pmf(self, k):
        """Calculates PDF value for a number of successes k."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        coefficient = (self.factorial(self.n) /
                       (self.factorial(k) * self.factorial(self.n - k)))

        pmf_res = coefficient * (self.p ** k) * ((1 - self.p) ** (self.n - k))

        return pmf_res

    def cdf(self, k):
        """Calculates CDF for a number of successes k."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        cdf_res = 0
        for i in range(k + 1):
            cdf_res += self.pmf(i)

        return cdf_res
