#!/usr/bin/env python3
"""
 This module contains the function add_matrices.
"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices."""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        new_matrix = [
            [mat1[row][i] + mat2[row][i] for i in range(len(mat1[0]))]
            for row in range(len(mat1))
        ]
        return new_matrix
    return None
