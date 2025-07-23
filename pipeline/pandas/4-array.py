#!/usr/bin/env python3
"""array selects values on DataFrame and converts it into numpy.ndarray"""


def array (df):
    """selects values on DataFrame and converts into numpy.ndarray"""
    df = df.loc[:, ['High', 'Close']].tail(10)
    np_ndarray = df.to_numpy()
    return np_ndarray
