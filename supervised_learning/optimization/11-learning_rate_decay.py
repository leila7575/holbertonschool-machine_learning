#!/usr/bin/env python3
"""This module contains the function learning_rate_decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Updates learning rate with inverse time decay."""
    step = global_step // decay_step
    updated_alpha = alpha / (1 + decay_rate * step)
    return updated_alpha
