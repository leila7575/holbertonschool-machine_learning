#!/usr/bin/env python3
"""This module contains the function train_model."""


import tensorflow.keras as K


def train_model(
    network, data, labels, batch_size, epochs, validation_data=None,
    early_stopping=False, patience=0, verbose=True, shuffle=False
):
    """Trains the model with mini-batch gradient descent."""
    if validation_data is not None:
        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience
        )
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1 if verbose else 0,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=[early_stopping]
        )
    return history
