#!/usr/bin/env python3
"""
 This module contains the function add_matrices.
"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices."""
    if (len(mat1) == len(mat2)) and 
    (len(mat1[i]) == len(mat2[i])) and
    (len(mat1[i][j]) == len(mat2[i][j])):
        new_matrix = [
            [[mat1[d][r][c] + mat2[d][r][c]
             for c in range(len(mat1[d][r]))]
            for r in range(len(mat1[d]))
        ]
        for d in range(len(mat1))
        ]
        return new_matrix
    return None
