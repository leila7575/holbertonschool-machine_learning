#!/usr/bin/env python3
"""contains class RNNDecoder"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder for machine translation"""
    def __init__(self, vocab, embedding, units, batch):
        """class constructor for RNN Decoder"""
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab,
            output_dim=embedding,
        )
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )
        self.F = tf.keras.layers.Dense(units=vocab)
        self.SelfAttention = SelfAttention(self.units)

    def call(self, x, s_prev, hidden_states):
        """call method for GRU forward pass"""
        x = self.embedding(x)
        context, _ = self.SelfAttention(s_prev, hidden_states)
        context = tf.expand_dims(context, 1)
        concatenated_vect = tf.concat([context, x], axis=-1)
        output, s = self.gru(concatenated_vect, initial_state=s_prev)
        output = tf.reshape(output, (output.shape[0], -1))
        y = self.F(output)
        return y, s
