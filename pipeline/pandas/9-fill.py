#!/usr/bin/env python3
"""Fills missing values"""


def fill(df):
    """Fill missing values on dataframe"""
    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df['Close'].ffill()
    for column in ['High', 'Low', 'Open']:
        df[column] = df[column].fillna(df['Close'])
    df[['Volume_(BTC)', 'Volume_(Currency)']] = (
        df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
    )
    return df
