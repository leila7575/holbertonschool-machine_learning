#!/usr/bin/env python3
"""Contains class Normal."""


class Normal:
    """Represents a normal distribution."""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Class constructor for normal distribution class."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            var = 0
            for i in range(len(data)):
                var += ((data[i] - self.mean) ** 2) / (len(data))
            self.stddev = var ** (1/2)

    def z_score(self, x):
        """Calculates z_score of a value x."""
        z_score_res = (x - self.mean) / self.stddev
        return z_score_res

    def x_value(self, z):
        """Calculates x-value of a given z-score."""
        x_value_res = z * self.stddev + self.mean
        return x_value_res

    def pdf(self, x):
        """Calculates PDF value for a value x."""

        pdf_res = (
            (1 / (self.stddev * (2 * 3.1415926536) ** 0.5)) *
            (2.7182818285 ** (-((x - self.mean) ** 2) /
                              (2 * (self.stddev ** 2))))
        )

        return pdf_res

    def cdf(self, x):
        """Calculates CDF value for a value x."""
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        erf = (
            (2 / (3.1415926536 ** 0.5)) *
            (z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42 + (z ** 9) / 216)
        )

        cdf_res = 0.5 * (1 + erf)
        return cdf_res
