#!/usr/bin/env python3
"""Data augmentation, adjusting contrast of an image"""


import tensorflow as tf


def change_contrast(image, lower, upper):
    """adjusts contrast of image for data augmentation"""
    adjusted_image = tf.image.random_contrast(image, lower, upper)
    return adjusted_image
