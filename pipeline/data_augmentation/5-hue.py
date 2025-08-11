#!/usr/bin/env python3
"""Data augmentation, changing hue of an image"""


import tensorflow as tf


def change_hue(image, delta):
    """changes hue of image for data augmentation"""
    adjusted_image = tf.image.adjust_hue(image, delta)
    return adjusted_image
