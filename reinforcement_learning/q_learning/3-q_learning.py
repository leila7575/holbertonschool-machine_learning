#!/usr/bin/env python3
"""Q learning training loop"""


import numpy as np
import gymnasium as gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(
    env, Q, episodes=5000, max_steps=100, alpha=0.1,
    gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05
):
    """Performs Q learning"""
    total_rewards = []
    for episode in range(episodes):
        episode_reward = 0
        state, info = env.reset()
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, info = env.step(action)

            if terminated and reward != 1:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )
            episode_reward += reward
            state = new_state
            if terminated or truncated:
                break
        total_rewards.append(episode_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    return Q, total_rewards
