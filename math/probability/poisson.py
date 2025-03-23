#!/usr/bin/env python3
"""Contains class Poisson."""


class Poisson:
    """Represents a Poisson distribution."""
    def __init__(self, data=None, lambtha=1.):
        """Class constructor for Poisson class."""
        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))
            
    def pmf(self, k):
        """Calculates PMF value for k successes."""
        if not isinstance(k, int):
            k = int(k)
            
        if k < 0:
            return 0
        
        factorial_k = 1
        for exp in range(2, k + 1):
            factorial_k *= exp
        
        pmf_res = (self.lambtha ** k) * (2.7182818285 **(-self.lambtha) / factorial_k)
        
        return pmf_res
    
    def cdf(self, k):
        """Calculates CDF value for k successes."""
        if not isinstance(k, int):
            k = int(k)
            
        if k < 0:
            return 0
        
        cdf_res = 0
        for i in range(k + 1):
            cdf_res += self.pmf(i)
            
        return cdf_res
        
