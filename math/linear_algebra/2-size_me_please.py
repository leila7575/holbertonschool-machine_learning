#!/usr/bin/env python3
"""
 This module contains the function matrix_shape.
"""
import numpy as np


def matrix_shape(matrix):
    """Calculates the shape of a matrix."""
    shape = np.array(matrix).shape
    shape_list = list(shape)
    return (shape_list)
