#!/usr/bin/env python3
"""
 This module contains the function mat_mul.
"""


def mat_mul(mat1, mat2):
    """Multiplies two matrices."""
    if len(mat1[0]) == len(mat2):
        new_matrix = [
            [0 for _ in range(len(mat2[0]))]
            for _ in range(len(mat1))
            ]
        for r in range(len(mat1)):
            for c in range(len(mat2[0])):
                for i in range(len(mat2)):
                    new_matrix[r][c] += mat1[r][i] * mat2[i][c]
        return new_matrix
    return None
