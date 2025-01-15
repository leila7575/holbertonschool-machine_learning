#!/usr/bin/env python3
"""This module contains the function calculate_accuracy."""


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction."""
    labels = tf.argmax(y, axis=1)
    predictions = tf.argmax(y_pred, axis=1)
    correct_predictions = tf.equal(labels, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
