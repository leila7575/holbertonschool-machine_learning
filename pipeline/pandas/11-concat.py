#!/usr/bin/env python3
"""concatenates two dataframes"""


import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """concatenates two dataframes"""
    df1 = index(df1)
    df2 = index(df2)
    concat_df = pd.concat(
        [df2.loc[:'1417411920', :], df1],
        keys=['bitstamp', 'coinbase'],
        axis=0
    )
    return concat_df
