#!/usr/bin/env python3
"""index function reindexes the dataframe with timestamp column"""


def index(df):
    """Reindexing the dataframe with timestamp column"""
    df = df.set_index('Timestamp', drop='True')
    return df
