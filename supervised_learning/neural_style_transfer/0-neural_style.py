#!/usr/bin/env python3
"""class NST for neural style transfer."""

import numpy as np
import tensorflow as tf


class NST:
    """class NST for neural style transfer."""
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """class constructor for NST neural style transfer."""
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray")
        
        if style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must have shape (h, w, 3)")
        
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if alpha < 0 or not isinstance(alpha, (int, float)):
            raise TypeError("alpha must be a non-negative number")

        if beta < 0 or not isinstance(beta, (int, float)):
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales image and normalizes pixel values."""
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )
            
        image = np.asarray(image)
        image = image.astype('float32')
        image /= 255.0
        
        max_dim = 512
        h, w, _ = image.shape
        long = max(h, w)
        scale = max_dim/long
        h_new, w_new = int(h*scale), int(w*scale)
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        resized_image = tf.image.resize(
            image,
            (h_new, w_new),
            method=tf.image.ResizeMethod.BICUBIC
        )
        resized_image = tf.clip_by_value(resized_image, 0.0, 1.0)
        resized_image = tf.expand_dims(resized_image, axis=0)
        return resized_image
