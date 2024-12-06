#!/usr/bin/env python3
"""
 This module contains the function cat_matrices.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two matrices."""
    if axis == 1:
        new_matrix = [mat1[i] + mat2[i] for i in range(len(mat1))]
    if axis == 0:
        new_matrix = mat1 + mat2
    return new_matrix
