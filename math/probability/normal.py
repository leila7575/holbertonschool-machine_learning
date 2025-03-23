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
