#!/usr/bin/env python3
"""This module contains function l2_reg_gradient_descent."""


import tensorflow as tf


def l2_reg_cost(cost, model):
    """calculates cost by adding the cost without L2 regularization
    and the total cost with L2 regularization."""
    return cost + model.losses
