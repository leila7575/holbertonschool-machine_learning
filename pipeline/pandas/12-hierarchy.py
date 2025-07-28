#!/usr/bin/env python3
"""Concatenates two sliced dataframe objects,
rearranges indexes and sorts data."""


import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """Concatenates two sliced dataframe objects,
    rearranges indexes and sorts data."""
    df1 = index(df1)
    df2 = index(df2)
    concat_df = pd.concat(
        [
            df2.loc['1417411980':'1417417980', :],
            df1.loc['1417411980':'1417417980', :]
        ],
        keys=['bitstamp', 'coinbase'],
        axis=0
    )
    concat_df = concat_df.reorder_levels([1, 0])
    concat_df = concat_df.sort_index(axis=0, ascending=True)
    return concat_df
