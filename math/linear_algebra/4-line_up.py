#!/usr/bin/env python3
"""
 This module contains the function add_arrays.
"""


def add_arrays(arr1, arr2):
    """Adds two arrays."""
    if len(arr1) == len(arr2):
        new_array = []
        for i in range(len(arr1)):
            new_array.append(arr1[i] + arr2[i])
        return new_array
    return None
