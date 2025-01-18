#!/usr/bin/env python3
"""This module contains the function train_model."""


import tensorflow.keras as K


def train_model(
    network, data, labels, batch_size, epochs, verbose=True, shuffle=False
):
    """Trains the model with mini-batch gradient descent."""
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1 if verbose else 0,
        shuffle=shuffle
        )
    return history
