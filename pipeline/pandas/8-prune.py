#!/usr/bin/env python3
"""prune function filters data in pd DataFrame removing NaN values"""


def prune(df):
    """Removes NaN values from Close column in pd DataFrame"""
    filtered_df = df.dropna(subset='Close')
    return filtered_df
