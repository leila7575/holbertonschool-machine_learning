#!/usr/bin/env python3
"""This module contains the function optimize_model."""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Sets up Adam optimization for the model"""
    network.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=alpha, beta_1=beta1, beta_2=beta2
            ),
        loss=K.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
        )
    return network
