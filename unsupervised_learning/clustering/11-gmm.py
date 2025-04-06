#!/usr/bin/env python3
"""Contains gmm function."""


import sklearn.mixture


def gmm(X, k):
    """computes GMM from dataset."""
    n, d = X.shape
    gmm = sklearn.mixture.GaussianMixture(n_components=k)
    gmm.fit(X)
    pi = gmm.weights_
    m = gmm.means_
    s = gmm.covariances_
    clss = gmm.predict(X)
    log_likelihood = sum(gmm.score_samples(X))
    n_components = gmm.n_components
    number_parameters = n_components * (d + d * (d + 1) / 2 + 1)
    log = len([i for i in range(n)])
    bic = log * number_parameters - 2 * log_likelihood
    return pi, m, s, clss, bic
