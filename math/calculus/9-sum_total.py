#!/usr/bin/env python3
"""
 This module contains the function summation_i_squared(n).
"""


def summation_i_squared(n, i=1):
    """Calculates the sigma sum of i squared"""
    if isinstance(n, int) and n >= 1:
        return (n * (n + 1) * (2 * n + 1)) // 6
    return None
