#!/usr/bin/env python3
"""Generate descriptive statistics of pd dataframe"""


def analyze(df):
    """Generates descriptive statistics excluding Timestamp column"""
    stat_df = df.drop(columns=['Timestamp']).describe()
    return stat_df
