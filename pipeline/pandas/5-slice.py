#!/usr/bin/env python3
"""slice extracts columns and rows from Dataframe"""


import pandas as pd


def slice(df):
    """Extracts columns and rows from Dataframe"""
    rows_index = []
    for index, row in df.iterrows():
        if index % 60 == 0:
            rows_index.append(index)
    sliced_df = df.loc[rows_index, ['High', 'Low', 'Close', 'Volume_(BTC)']]
    return sliced_df
