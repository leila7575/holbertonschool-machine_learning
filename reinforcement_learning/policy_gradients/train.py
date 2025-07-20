#!/usr/bin/env python3
"""
Training loop implementation with monte carlo policy gradient
"""


import numpy as np
import matplotlib.pyplot as plt
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Training loop implementation with monte carlo policy gradient"""
    scores = []
    states = env.observation_space.shape[0]
    actions = env.action_space.n
    weights = np.random.randn(states, actions)
    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_info = []

        while True:
            if show_result and episode % 1000 == 0:
                env.render()
            action, gradient = policy_gradient(state, weights)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_info.append((state, action, reward, gradient))
            if terminated or truncated:
                break
            state = next_state

        G = 0
        Gt = []
        for i in range(len(episode_info)-1, -1, -1):
            state, action, reward, gradient = episode_info[i]
            G = reward + gamma * G
            Gt.append(G)

        Gt.reverse()
        Gt = np.array(Gt)

        for i in range(len(episode_info)):
            state, action, reward, gradient = episode_info[i]
            weights += alpha * gradient * Gt[i]

        score = sum([ep[2] for ep in episode_info])
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    plt.plot(scores)
    plt.show()

    return scores
