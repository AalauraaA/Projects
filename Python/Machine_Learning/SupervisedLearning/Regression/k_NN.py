# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:19:57 2020

@author: Laura
------------------------------------------------------------------------------
SUPERVISED LEARNING - REGRESSION
------------------------------------------------------------------------------
Supervised learning is to build a model on training data and then be able to 
make accurate prediction on new, unseen data with the same characteristics as
the training data. If it make accurate prediction the model is generalize, and
we want to have a model to generalize as accurately as possible.
Building a model to complex lead to the model being overfitted but building 
the model to simple lead to the model being underfitted.
------------------------------------------------------------------------------

------------------------------------------------------------------------------
k-Nearest Neighbour (k-NN) for Regression
------------------------------------------------------------------------------
k-NN is a simple ML algorithm as only consist of storing the training dataset.
k-NN then make a prediction for a new data point by finding the closest data 
points in the training data set (the nearest neighbour).

In its simple version the k-NN algorithm only consider one nearest neighbour, 
the closest training point. The prediction is then the known output for the 
closest training point.

When considering k nearest neighbours (more than one), we then use an average,
or mean, of the relevant neighbours to assign a class label. 

In practice a small number of neighbours works well.

Strenght:
    Easy to understand
    Give reasonable performance without a lot o adjustments
    Good baseline method
Weakness:
    Larger training dataset lead to slower predictions
    Does not perform well on datasets with many features or sparse datasets
"""
# =============================================================================
# Libraries Import
# =============================================================================
import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor

# =============================================================================
# Data Import
# =============================================================================
" Synthetic Wave Data Set "
X, y = mglearn.datasets.make_wave(n_samples=60) # generate dataset
print("X.shape: {}".format(X.shape))

# plot dataset
plt.plot(X, y, 'o')                             
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")

" Real World Regression Data Set - Boston Housing "
#task is to predict the median value of homes in several neighbourhoods in
#1970s, using crime rate, proximity to the Charles River, highway accessbility
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
#contain 506 data points and 13 features
X_boston, y_boston = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X_boston.shape))

# =============================================================================
# Synthetic Wave Dataset
# =============================================================================
" k = 1 "
mglearn.plots.plot_knn_regression(n_neighbors=1)

" k = 3 "
mglearn.plots.plot_knn_regression(n_neighbors=3)

" Scikit-learn "
#training and test data from wave dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3) # k-NN with k = 3
reg.fit(X_train, y_train) # fit the classifier (storing the dataset)

#for each test data point in the test dataset its compute the nearest neighbour
print("Test set predictions:\n{}".format(reg.predict(X_test)))
#Test set predictions:
#[-0.054 0.357 1.137 -1.894 -1.139 -1.631 0.357 0.912 -0.447 -1.139]

#to evaluate how well the model generalize, we use the score method, which
#for regressor returns a R2 score (coefficient of determination) which yields
#a score between 1 and 0
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))
#Test set R^2: 0.83
#the k-NN model is 86% accurate meaning that the model predict the class label
#correctly for 86% of the samples in the test dataset.

" Analyzing with varying k "
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

line = np.linspace(-3, 3, 1000).reshape(-1, 1) # create 1,000 data points, evenly spaced between -3 and 3

for n_neighbours, ax in zip([1, 3, 9], axes):
    #make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbours)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title(
    "{} neighbour(s)\n train score: {:.2f} test score: {:.2f}".format(
    n_neighbours, reg.score(X_train, y_train),
    reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

axes[0].legend(["Model predictions", "Training data/target",
"Test data/target"], loc="best")

"""
Using only a single neighbour leads to a very unsteady prediction. Considering 
more neighbours leads to smoother predictions, but these do not fit the 
training data as well.
"""