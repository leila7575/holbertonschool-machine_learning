#!/usr/bin/env python3
"""Contains the function lenet5."""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """Builds a lenet5 convolutional network"""
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
        )(x)
    pool1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv1)
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )(pool1)
    pool2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
    )(conv2)
    flatten = tf.layers.flatten(pool2)
    dense1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )(flatten)
    dense2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )(dense1)
    logits = tf.layers.Dense(
        units=10,
        activation=None,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0)
    )(dense2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=y,
        logits=logits
    )
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    labels = tf.argmax(y, axis=1)
    predictions = tf.argmax(logits, axis=1)
    correct_predictions = tf.equal(labels, predictions)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return logits, optimizer, loss, accuracy
