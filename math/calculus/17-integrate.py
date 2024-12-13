#!/usr/bin/env python3
"""
 This module contains the function poly_integral.
"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial."""
    if not isinstance(poly, list) or not isinstance(C, int) or len(poly) == 0:
        return None
    if poly == [0]:
        return [0, 7]
    integral = [C]
    for exp, coefficient in enumerate(poly):
        integral_coefficient = coefficient / (exp + 1)
        if integral_coefficient.is_integer():
            integral.append(int(integral_coefficient))
        else:
            integral.append(integral_coefficient)
    return integral
