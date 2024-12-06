#!/usr/bin/env python3
"""
 This module contains the function np_cat.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concatenates matrices"""
    return np.concatenate((mat1, mat2), axis)
