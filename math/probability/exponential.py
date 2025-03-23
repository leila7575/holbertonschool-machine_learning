#!/usr/bin/env python3
"""Contains class Exponential."""


class Exponential:
    """Represents an exponential distribution."""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor for expoential class."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """Calculates PDF value for a time period x."""
        if x < 0:
            return 0

        pdf_res = self.lambtha * (
            2.7182818285 ** (-self.lambtha * x)
        )

        return pdf_res

    def cdf(self, x):
        """Calculates CDF value for period time x."""
        if x < 0:
            return 0

        cdf_res = 1 - 2.7182818285 ** (-self.lambtha * x)

        return cdf_res
