#!/usr/bin/env python3
"""Contains kmeans function."""


import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    n, d = X.shape

    min_values = np.min(X, axis=0)
    max_values = np.max(X, axis=0)
    c = np.random.uniform(min_values, max_values, size=(k, d))

    iteration = 0
    prev_centroids = None

    while iteration < iterations:
        clss = np.zeros(n, dtype=int)
        data_points = [[] for _ in range(k)]
        for i in range(n):
            euclidean_dist = np.sqrt(np.sum((X[i] - c)**2, axis=1))
            centroid_index = np.argmin(euclidean_dist)
            clss[i] = centroid_index
            data_points[centroid_index].append(X[i])

        prev_centroids = c.copy()

        for j in range(k):
            if len(data_points[j]) == 0:
                c[j] = np.random.uniform(min_values, max_values, size=(d, ))
            else:
                c[j] = np.mean(data_points[j], axis=0)

        if np.all(c == prev_centroids):
            break

        iteration += 1

    return c, clss
