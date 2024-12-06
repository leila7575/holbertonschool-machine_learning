#!/usr/bin/env python3
"""
 This module contains the function np_cat.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates matrices"""
    if axis == 1:
        concatenated_array = np.column_stack((mat1, mat2))
    if axis == 0:
        concatenated_array = np.concatenate((mat1, mat2))
    return concatenated_array
