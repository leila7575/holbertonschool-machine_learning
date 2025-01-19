#!/usr/bin/env python3
"""This module contains the function train_model."""


import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                learning_rate_decay=False,
                alpha=0.1,
                decay_rate=1,
                save_best=False,
                filepath=None,
                verbose=True,
                shuffle=False
                ):
    """Trains the model with mini-batch gradient descent."""
    callbacks = []
    if validation_data is not None:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience
            )
            callbacks.append(early_stop)
        if learning_rate_decay:
            learning_rate_schedule = (
                K.optimizers.schedules.InverseTimeDecay(
                    initial_learning_rate=alpha,
                    decay_steps=1,
                    decay_rate=decay_rate,
                    staircase=True
                )
            )
            network.optimizer.learning_rate = learning_rate_schedule

            class PrintCallback(K.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    learning_rate = alpha / (1 + decay_rate * epoch)
                    print(
                        f'Epoch {epoch + 1}: '
                        f'LearningRateScheduler setting learning rate to '
                        f'{learning_rate}. '
                    )

            callbacks.append(PrintCallback())

            if save_best and filepath is not None:
                model_checkpoint = K.callbacks.ModelCheckpoint(
                    filepath=filepath,
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=verbose
                )
                callbacks.append(model_checkpoint)

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
