#!/usr/bin/env python3
"""Renames column and converts values of DataFrame"""


import pandas as pd


def rename(df):
    """Renames Timestamp column, converts values to datetime
    and selects columns."""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    df = df.loc[:, ["Datetime", "Close"]]
    return df
