#!/usr/bin/env python3
"""This module contains the function create_RMSProp_op."""


import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """sets up gradient descent with RMSProp optimization algorithm."""
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon
    )
    return optimizer
