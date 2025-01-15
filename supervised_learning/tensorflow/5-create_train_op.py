#!/usr/bin/env python3
"""This module contains the function create_train_op."""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_train_op(loss, alpha):
    """Returns the training operation using gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=alpha
        ).minimize(loss)
    return optimizer
