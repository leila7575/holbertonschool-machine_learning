#!/usr/bin/env python3
"""Data augmentation, rotating an image"""


import tensorflow as tf


def rotate_image(image):
    """rotates image for data augmentation"""
    rotated_image = tf.image.rot90(image, k=1)
    return rotated_image
