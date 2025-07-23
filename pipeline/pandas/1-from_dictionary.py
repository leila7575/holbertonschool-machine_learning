#!/usr/bin/env python3
"""creating DataFrame from dictionary"""


import pandas as pd


df = pd.DataFrame(
    {
        "First": [0.0, 0.5, 1.0, 1.5],
        "Second": pd.Categorical(['one', 'two', 'three', 'four'])
    }, index=list("ABCD")
)
