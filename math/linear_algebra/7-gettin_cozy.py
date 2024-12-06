#!/usr/bin/env python3
"""
 This module contains the function cat_matrices.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices."""
    new_matrix = []
    if axis == 1:
        if len(mat1) == len(mat2):
            new_matrix = [mat1[i] + mat2[i] for i in range(len(mat1))]
            return new_matrix
        return None
    if axis == 0:
        if len(mat1[0]) == len(mat2[0]):
            new_matrix = mat1 + mat2
            return new_matrix
        return None
    return None
