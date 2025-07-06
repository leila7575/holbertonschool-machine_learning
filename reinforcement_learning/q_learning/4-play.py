#!/usr/bin/env python3
"""contains play function that has the agent play one episode"""


import numpy as np
import gymnasium as gym
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def play(env, Q, max_steps=100):
    """Has the trained agent play one episode"""
    total_reward = 0
    rendered_outputs = []
    state, info = env.reset()
    rendered_outputs.append(env.render())
    for step in range(max_steps):
        action = np.argmax(Q[state][:])
        new_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rendered_outputs.append(env.render())
        state = new_state
        if terminated or truncated:
            break
    return total_reward, rendered_outputs
