#!/usr/bin/env python3
"""This module contains the function create_mini_batches."""


def moving_average(data, beta):
    """Calculates the weighted moving average."""
    corrected_moving_average = []
    vt = 0
    for i, val in enumerate(data):
        vt = beta * vt + (1 - beta) * val
        vt_unbiased = vt / (1 - beta**(i + 1))
        corrected_moving_average.append(vt_unbiased)
    return corrected_moving_average
