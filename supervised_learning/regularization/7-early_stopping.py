#!/usr/bin/env python3
"""This module contains the function early_stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines when to stop gradient descent
    based on validation cost and a threshold."""
    cost_diff = opt_cost - cost
    if cost_diff > threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return (True, count)
    return (False, count)
