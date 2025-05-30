#!/usr/bin/env python3
"""Class SelfAttention"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """computes attention for machine translation"""
    def __init__(self, units):
        """class constructor for attention class"""
        super().__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Applies self attention bahdanau mechanisms
        return context vector and attention weights"""
        s_prev = tf.expand_dims(s_prev, 1)
        W = self.W(s_prev)
        U = self.U(hidden_states)
        sum_W_U = tf.nn.tanh(W + U)
        score = self.V(sum_W_U)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
