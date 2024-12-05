#!/usr/bin/env python3
"""
 This module contains the function matrix_transpose.
"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix."""
    new_matrix = [
        [matrix[r][c] for r in range(len(matrix))]
        for c in range(len(matrix[0]))
        ]
    return new_matrix
