#!/usr/bin/env python3
"""This module contains the function train_model."""


import tensorflow.keras as K


def train_model(
    network, data, labels, batch_size, epochs, validation_data=None,
    early_stopping=False, patience=0, verbose=True, shuffle=False
):
    """Trains the model with mini-batch gradient descent."""
    callbacks = []
    if validation_data is not None:
        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience
        )
        callbacks.append(early_stopping)
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=callbacks
    )
    return history
