# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:38:11 2020

@author: Laura

Recommendation System in Python
"""
import pandas as pd
import numpy as np
import sklearn

# =============================================================================
# Content-Based Recommendation System Based on Machine Learning
# =============================================================================
" Nearest Neighbors Algorithm "
from sklearn.neighbors import NearestNeighbors

# Import of data
cars = pd.read_csv('mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']
print(cars.head())
print("")

# The wanted specification for a car. I will try to used the model to recommendate
# a car similiar to the specifications.
t = [15, 300, 160, 3.2]

# Select some data from the full data set
X = cars.iloc[:,[1, 3, 4, 6]].values
print(X[0:5])
print("")

# Lets used the nearest neughbours model on the select data
nbrs = NearestNeighbors(n_neighbors=1).fit(X)
print(nbrs.kneighbors([t]))
print("")

# Lets look if the model find a car with have the specific wishes 't'
print(cars)
print('')

# I can see that the suggest car in entry 22 matches the wanted specifications






