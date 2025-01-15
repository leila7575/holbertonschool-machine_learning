#!/usr/bin/env python3
"""This module contains the function calculate_loss."""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_loss(y, y_pred):
    """Calculates the softmax cross entropy loss of a prediction."""
    with tf.name_scope('softmax_cross_entropy_loss'):
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
        average_loss = tf.reduce_mean(cross_entropy_loss, name='value')
    return average_loss