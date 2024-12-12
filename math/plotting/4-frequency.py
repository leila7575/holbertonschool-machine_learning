#!/usr/bin/env python3
"""
This module contains the function frequency.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Defines a histogram of student grades"""
    array = np.arange(0, 101, 10)
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.xlim(0, 100)
    plt.xticks(array)
    plt.ylim(0, 30)
    plt.title("Project A")
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.hist(student_grades, edgecolor='black', bins=array)
    plt.show()
