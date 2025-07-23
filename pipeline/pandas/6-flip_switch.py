#!/usr/bin/env python3
"""flip_switch sorts Dataframe data in reverse chronological order"""


def flip_switch(df):
    """sorts Dataframe data in reverse chronological order"""
    sorted_df = df.sort_values(['Timestamp'], ascending=False).T
    return sorted_df
