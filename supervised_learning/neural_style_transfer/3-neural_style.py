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
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

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
        self.model = self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales image and normalizes pixel values."""
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        max_dim = 512
        h, w, _ = image.shape
        long = max(h, w)
        scale = max_dim/long
        h_new, w_new = int(h*scale), int(w*scale)
        if np.max(image) > 1.0:
            image = image / 255.0
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        resized_image = tf.image.resize(
            image,
            (h_new, w_new),
            method=tf.image.ResizeMethod.BICUBIC
        )
        resized_image = tf.clip_by_value(resized_image, 0.0, 1.0)
        resized_image = tf.expand_dims(resized_image, axis=0)
        return resized_image

    def load_model(self):
        """Creates the model."""
        base_model = tf.keras.applications.VGG19(
            weights='imagenet', include_top=False
        )
        base_model.trainable = False
        style_outputs = [
            base_model.get_layer(name).output for name in self.style_layers
        ]
        content_outputs = [base_model.get_layer(self.content_layer).output]
        model_output = style_outputs + content_outputs
        model = tf.keras.models.Model(base_model.input, model_output)
        return model

    @staticmethod
    def gram_matrix(input_layer):
        """calculates the gram matrix of an input tensor."""
        if not isinstance(
            input_layer, (tf.Tensor, tf.Variable)
        ) or tf.rank(input_layer) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")
        gram_result = tf.linalg.einsum(
            'bijc, bijd->bcd', input_layer, input_layer
        )
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return gram_result/num_locations

    def generate_features(self):
        """Extracts style and content features for style cost calculation."""
        style_outputs = self.model(self.style_image)
        content_outputs = self.model(self.content_image)

        style_features = style_outputs[:len(self.style_layers)]
        content_feature = content_outputs[-1][0]
        gram_style_features = [
            self.gram_matrix(style_feature) for style_feature in style_features
        ]

        self.content_feature = content_feature
        self.gram_style_features = gram_style_features
