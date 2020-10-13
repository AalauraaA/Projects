# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 19:21:01 2020

@author: Laura
------------------------------------------------------------------------------
Data Visualization in Python - Pie Chart Plot
------------------------------------------------------------------------------
If we want to display elements of a data set as proportions of a whole, we can 
use a pie chart. 

The commando are:
    - plt.pie(labels=, autopct=)
    - plt.axis('equal')
"""
import matplotlib.pyplot as plt

budget_data = [500, 1000, 750, 300, 100]
budget_categories = ['marketing', 'payroll', 'engineering', 'design', 'misc']

plt.pie(budget_data)
plt.axis('equal')
plt.legend(budget_categories, autopct='%0.1f%%')
plt.show()