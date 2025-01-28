#!/usr/bin/env python3
"""This module contains the fuction create_confusion_matrix."""


import numpy as np


def create_confusion_matrix(labels, logits):
    """Creates a confusion matrix.

    Args:
        labels: one-hot encoded array with the correct labels.
        logits: one-hot encoded array with predicted labels.

    Return:
        Confusion matrix.
    """
    labels_classes = np.argmax(labels, axis=1)
    logits_classes = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    matrix = np.zeros((classes, classes))
    m = labels.shape[0]
    for i in range(classes):
        for j in range(classes):
            for k in range(m):
                if labels_classes[k] == i and logits_classes[k] == j:
                    matrix[i][j] = matrix[i][j] + 1
    return matrix
