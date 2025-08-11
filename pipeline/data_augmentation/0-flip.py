#!/usr/bin/env python3
"""Data augmentation, flipping an image horizontally"""


import tensorflow as tf


def flip_image(image):
    """flips image horizontally"""
    flipped_image = tf.image.flip_left_right(image)
    return flipped_image
