#!/usr/bin/env python3
"""
This module contains the function all_in_one.
"""
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """Defines a figure composed by a 3x2 grid of subplots with 5 figures."""

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure(figsize=(6.4, 4.8))
    plt.suptitle("All in One")

    plt.subplot(3, 2, 1)
    array_y1 = np.arange(0, 1001, 500)
    x0 = np.arange(0, 10)
    plt.xlim(0, 10)
    plt.xticks(fontsize='x-small')
    plt.yticks(array_y1, fontsize='x-small')
    plt.plot(range(11), y0, linestyle='solid', color='red')

    plt.subplot(3, 2, 2)
    plt.title("Men's Height vs Weight", fontsize='x-small')
    plt.xlabel("Height (in)", fontsize='x-small')
    plt.ylabel("Weight (lbs)", fontsize='x-small')
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')
    plt.scatter(x1, y1, color='magenta')

    plt.subplot(3, 2, 3)
    plt.yscale("log")
    plt.xlim(0, 28650)
    plt.title("Exponential Decay of C-14", fontsize='x-small')
    plt.xlabel("Time (years)", fontsize='x-small')
    plt.ylabel("Fraction Remaining", fontsize='x-small')
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')
    plt.plot(x2, y2)

    plt.subplot(3, 2, 4)
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.title("Exponential Decay of Radioactive Elements", fontsize='x-small')
    plt.xlabel("Time (years)", fontsize='x-small')
    plt.ylabel("Fraction Remaining", fontsize='x-small')
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')
    plt.plot(x3, y31, linestyle='--', color='red')
    plt.plot(x3, y32, linestyle='-', color='green')
    plt.legend(['C-14', 'Ra-226'], loc='upper right', fontsize='x-small')

    plt.subplot(3, 2, (5, 6))
    array_x4 = np.arange(0, 101, 10)
    array_y4 = np.arange(0, 31, 10)
    plt.xlim(0, 100)
    plt.xticks(array_x4, fontsize='x-small')
    plt.yticks(array_y4, fontsize='x-small')
    plt.ylim(0, 30)
    plt.title("Project A", fontsize='x-small')
    plt.xlabel("Grades", fontsize='x-small')
    plt.ylabel("Number of Students", fontsize='x-small')
    plt.hist(student_grades, edgecolor='black', bins=array_x4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
