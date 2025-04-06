#!/usr/bin/env python3
"""Contains agglomerative function."""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """performs agglomerative clustering on a dataset"""
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')

    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(
        linkage, t=dist, criterion='distance'
    )
    plt.show()

    return clss
