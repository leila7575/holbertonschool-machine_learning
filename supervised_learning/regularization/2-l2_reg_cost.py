#!/usr/bin/env python3
"""This module contains function l2_reg_gradient_descent."""


import tensorflow as tf


def l2_reg_cost(cost, model):
    return cost + model.losses
