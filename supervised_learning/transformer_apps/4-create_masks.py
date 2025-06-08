#!/usr/bin/env python3
"""Creates all masks for training/validation in a transformer model"""


import tensorflow as tf


def create_padding_mask(seq):
    """Creates padding mask"""
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """Creates a look-ahead mask"""
    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def create_masks(inputs, target):
    """Creates all masks for training/validation in a transformer model"""
    encoder_mask = create_padding_mask(inputs)
    decoder_mask = create_padding_mask(inputs)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    target_padding_mask = create_padding_mask(target)
    target_padding_mask = tf.squeeze(target_padding_mask, axis=2)

    combined_mask = tf.maximum(look_ahead_mask, target_padding_mask)

    return encoder_mask, combined_mask, decoder_mask
