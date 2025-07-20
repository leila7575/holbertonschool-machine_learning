#!/usr/bin/env python3
"""
Contains function policy and policy_gradient
Computes Monte Carlo policy gradient
"""


import numpy as np


def softmax(x):
    """Applies softmax function"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def policy(matrix, weight):
    """computes the policy with a weight of a matrix"""
    logits = np.dot(matrix, weight)
    softmax_prob = softmax(logits)
    return softmax_prob


def policy_gradient(state, weight):
    """Computes Monte Carlo policy gradient"""
    softmax_prob = policy(state, weight)
    actions_number = softmax_prob.shape[1]
    action = np.random.choice(actions_number, p=softmax_prob[0])
    one_hot_encoding = np.eye(actions_number)[action]
    gradient = np.outer(state, one_hot_encoding - softmax_prob[0])
    return action, gradient
