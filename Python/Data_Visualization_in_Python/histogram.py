# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:22:29 2020

@author: Laura
------------------------------------------------------------------------------
Data Visualization in Python - Histogram
------------------------------------------------------------------------------
Sometimes we want to get a feel for a large dataset with many samples beyond 
knowing just the basic metrics of mean, median, or standard deviation. To get 
more of an intuitive sense for a dataset, we can use a histogram to display 
all the values. A histogram tells us how many values in a dataset fall between 
different sets of numbers. All bins in a histogram are always the same size. 
The width of each bin is the distance between the minimum and maximum values 
of each bin. Each bin is represented by a different rectangle whose height is '
the number of elements from the dataset that fall within that bin.

Another problem we face is that our histograms might have different numbers of 
samples, making one much bigger than the other. We can normalize our histograms 
using normed=True. This command divides the height of each column by a constant 
such that the total shaded area of the histogram sums to 1.

The commandos are:
    - plt.hist(bins=, range=, alpha=, histtype='step', size=, normed=)
"""
import matplotlib.pyplot as plt
import numpy as np

a = np.random.normal(loc=64, scale=2, size=10000)
b = np.random.normal(loc=70, scale=2, size=100000)

plt.hist(a, range=(55, 75), bins=20, alpha=0.5, normed=True)
plt.hist(b, range=(55, 75), bins=20, alpha=0.5, normed=True)
plt.show()
