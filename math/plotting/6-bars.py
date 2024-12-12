#!/usr/bin/env python3
"""
This module contains the function bars.
"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Defines a stacked bar graph
    representing the number of fruits for each person per type of fruit."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))
    x = ['Farrah', 'Fred', 'Felicia']
    apples, bananas, oranges, peaches = fruit
    plt.bar(x, apples, width=0.5, color='red', label='apples')
    plt.bar(x, bananas, width=0.5, bottom=apples, color='yellow')
    plt.bar(x, oranges, width=0.5, bottom=apples+bananas, color='#ff8000')
    plt.bar(
        x,
        peaches,
        width=0.5,
        bottom=apples+bananas+oranges,
        color='#ffe5b4'
        )
    plt.yticks(np.arange(0, 81, 10))
    plt.ylabel("Quantity of Fruit")
    plt.legend(['apples', 'bananas', 'oranges', 'peaches'])
    plt.title("Number of Fruit per Person")
    plt.show()
