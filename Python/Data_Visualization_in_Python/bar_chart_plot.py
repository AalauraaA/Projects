# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:19:18 2020

@author: Laura
------------------------------------------------------------------------------
Data Visualization in Python - Bar Chart Plot
------------------------------------------------------------------------------
The 'plt.bar' function allows you to create simple bar charts to compare 
multiple categories of data. You call plt.bar with two arguments:
    * the x-values — a list of x-positions for each bar
    * the y-values — a list of heights for each bar

We can use a bar chart to compare two sets of data with the same types of axis
values. To do this, we plot two sets of bars next to each other, so that the 
values of each category can be compared.     

The commands are:
    - plt.bar(bottom=, yerr=, capsize=)
"""
import matplotlib.pyplot as plt

drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 =  [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]

n = 1   # This is our first dataset (out of 2)
t = 2   # Number of datasets
d = 6   # Number of sets of bars
w = 0.8 # Width of each bar
store1_x = [t*element + w*n for element
             in range(d)]

plt.bar(store1_x, sales1)

n = 2   # This is our second dataset (out of 2)
t = 2   # Number of datasets
d = 6   # Number of sets of bars
w = 0.8 # Width of each bar
store2_x = [t*element + w*n for element
             in range(d)]

plt.bar(store2_x, sales2)
plt.show()

plt.bar(range(len(drinks)),
  sales1)
plt.bar(range(len(drinks)),
  sales2, bottom=sales1)

plt.legend(['Location 1', 'Location 2'])
plt.show()
ounces_of_milk = [6, 9, 4, 0, 9, 0]
error = [0.6, 0.9, 0.4, 0, 0.9, 0]
plt.bar(range(len(drinks)), ounces_of_milk, yerr=error, capsize=5)
plt.show()
