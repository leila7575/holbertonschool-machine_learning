#!/usr/bin/env python3
"""Data augmentation, adjusting brightness of an image"""


import tensorflow as tf


def change_brightness(image, max_delta):
    """adjusts brightness of image for data augmentation"""
    adjusted_image = tf.image.random_brightness(image, max_delta)
    return adjusted_image
