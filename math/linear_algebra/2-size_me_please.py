#!/usr/bin/env python3
"""
 This module contains the function matrix_shape.
"""


def matrix_shape(matrix):
    """Calculates the shape of a matrix."""
    if isinstance(matrix, list):
        return [len(matrix)] + matrix_shape(matrix[0])
    return []
