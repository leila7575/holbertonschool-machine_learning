#!/usr/bin/env python3
"""
 This module contains the function poly_derivative.
"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial."""
    derivative = []
    if not isinstance(poly, list):
        return None
    for i in range(1, len(poly)):
        derivative.append(poly[i] * i)
        if derivative == 0:
            return [0]
    return derivative
