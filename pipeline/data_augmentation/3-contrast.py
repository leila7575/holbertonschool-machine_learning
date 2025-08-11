#!/usr/bin/env python3
"""Data augmentation, adjusting contrast of an image"""


import tensorflow as tf


def change_contrast(image, lower, upper):
    """adjusts contrast of image for data augmentation"""
    contrast_factor = tf.random.uniform([], lower, upper, dtype=tf.float32)
    adjusted_image = tf.image.adjust_contrast(image, contrast_factor)
    return adjusted_image
