#!/usr/bin/env python3
"""Contains kmeans function."""


import sklearn.cluster


def kmeans(X, k):
    """performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)

    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
