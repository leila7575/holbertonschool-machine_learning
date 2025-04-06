#!/usr/bin/env python3
"""Contains initialize function for k-means
cluster centroids initialization."""


import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for k-means"""
    
    try:
        n, d = X.shape
        min_values = np.min(X, axis=0)
        max_values = np.max(X, axis=0)
        centroids = np.random.uniform(min_values, max_values, size=(k, d))
        return np.array(centroids)
    except Exception:
        return None
