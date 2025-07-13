#!/usr/bin/env python3
"""SARSA algorithm"""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Chooses next action based on epsilon-greedy policy."""
    p = np.random.uniform()
    number_actions = Q.shape[1]
    if p > epsilon:
        return np.argmax(Q[state][:])
    else:
        return np.random.randint(number_actions)


def sarsa_lambtha(
    env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
    gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """Updates Q values with SARSA algorithm"""
    first_epsilon = epsilon
    for episode in range(episodes):
        E = np.zeros(Q.shape)
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        for step in range(max_steps):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)
            delta = reward + (
                gamma * Q[next_state, next_action]
            ) - Q[state, action]
            E[state, action] += 1
            Q = Q + alpha * delta * E
            E = E * gamma * lambtha
            if terminated or truncated:
                break
            state, action = next_state, next_action
        decay_factor = np.exp(-epsilon_decay * episode)
        epsilon = min_epsilon + (first_epsilon - min_epsilon) * decay_factor

    return Q
