#!/usr/bin/env python3
"""This module contains the function create_momentum_op."""


import numpy as np
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """sets up gradient descent with momentum optimization algorithm."""
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
