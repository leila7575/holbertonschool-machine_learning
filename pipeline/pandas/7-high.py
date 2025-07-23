#!/usr/bin/env python3
"""high sorts Dataframe data by High price in descending order"""


def high(df):
    """sorts Dataframe data by High price in descending order"""
    sorted_df = df.sort_values(['High'], ascending=False)
    return sorted_df
