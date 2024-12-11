#!/usr/bin/env python3
"""
 This module contains the function chage_scale.
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Defines a line graph representing C14 exponential decay
    with the y axis logarythmically scaled."""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.yscale("log")
    plt.xlim(0, 28650)
    plt.title("Exponential Decay of C-14")
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.plot(x, y)
    plt.show()
