#!/usr/bin/env python3
"""
 This module contains the function line.
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """Defines a graph with y = x ^ 3 and x in range 0 to 10"""

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 10)
    plt.plot(range(11), y, linestyle='solid', color='red')
    plt.xlim(0, 10)
    plt.show()
