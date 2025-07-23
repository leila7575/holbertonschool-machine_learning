#!/usr/bin/env python3
"""Renames column and converts values of DataFrame"""


import pandas as pd


def rename(df):
    """Renames Timestamp column, converts values to datetime
    and selects columns."""
    new_df = df.rename(columns={"Timestamp": "Datetime"})
    new_df["Datetime"] = new_df["Datetime"].astype("datetime64[ns]")
    new_df = new_df.loc[:, ["Datetime", "Close"]]
    return new_df
