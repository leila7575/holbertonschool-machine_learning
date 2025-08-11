#!/usr/bin/env python3
"""Data augmentation, cropping an image"""


import tensorflow as tf


def crop_image(image, size):
    """crops image for data augmentation"""
    cropped_image = tf.image.random_crop(image, size)
    return cropped_image
