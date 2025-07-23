#!/usr/bin/env python3
"""loads data from file into DataFrame"""


import pandas as pd


def from_file(filename, delimiter):
    """loads data from file into DataFrame"""
    df = pd.read_csv(filename, delimiter=delimiter)
    return df
