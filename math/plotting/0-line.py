#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():

    y = np.arange(0, 10) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 10)
    plt.plot(x, y, color='red')
    plt.show()
