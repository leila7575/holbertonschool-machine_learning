#!/usr/bin/env python3
"""
Loads bitcoin data from coinbaseUSD csv file
Handles missing values, filters data from 2017 and beyond,
aggregates values to daily intervals and  plots the data
for bitcoin price trends visualization.
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns=['Weighted_Price'])
df = df.rename(columns={'Timestamp': 'Date'})
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df = df.set_index('Date', drop='True')
df['Close'] = df['Close'].ffill()
for column in ['High', 'Low', 'Open']:
    df[column] = df[column].fillna(df['Close'])
df[['Volume_(BTC)', 'Volume_(Currency)']] = (
    df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)
)
df = df['2017-01-01':]
df = df.groupby([pd.Grouper(freq='D')]).agg(
    {
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    }
)

print(df)

plot = df.plot()
plt.show()
