#!/usr/bin/env python3
"""contains class RNNEncoder"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN encoder for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """class constructor for RNN encoder"""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding,
            embeddings_initializer='uniform'
        )
        self.gru = tf.keras.layers.GRU(
            units,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )

    def initialize_hidden_state(self):
        """Initializes hidden states of RNN cell"""
        hidden_states = tf.zeros(
            (self.batch, self.units),
            dtype=tf.dtypes.float32
        )
        return hidden_states

    def call(self, x, initial):
        """call method for GRU forward pass"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
