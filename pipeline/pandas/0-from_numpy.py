#!/usr/bin/env python3
"""from_numpy creates panda dataframe from numpy ndarray"""


import pandas as pd


def from_numpy(array):
    """Creates panda dataframe from numpy ndarray"""
    column_index = [chr(char) for char in range(65, 91)]
    df = pd.DataFrame(array, columns=column_index[:array.shape[1]])
    return df
