#!/usr/bin/env python3
"""
 This module contains the function summation_i_squared(n).
"""


def summation_i_squared(n, i=1):
    """Calculates the sigma sum of i squared"""
    if isinstance(n, int) and n >= 1:
        if i > n:
            return 0
        return i ** 2 + summation_i_squared(n, i + 1)
    return None
