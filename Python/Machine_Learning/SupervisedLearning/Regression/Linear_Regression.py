# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:41:03 2020

@author: Laura

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
Linear Model for Regression
------------------------------------------------------------------------------
Linear models make a prediction using a linear function of the input features:
    y = a[0] * x[0] + a[1]*x[1] + ... + a[p] * x[p] + b
Here x[0]-x[p] is the features of a single data point, a and b are parameters
of the model that is learned. For a dataset with only one feature the linear
function look like this:
    y = a[0] * x[0] + b
which is the function for a line.

Linear models for regression can be characterized as regression models for 
which the prediction is a line for a single feature, a plane when using two 
features, or a hyperplane in higher dimensions (that is, when using more 
features).

For datasets with many features, linear models can be very powerful. If you 
have more features than training data points, any target y can be perfectly 
modeled (on the training set) as a linear function.

Linear regression, or ordinary least squares (OLS), is the simplest and most 
classic linear method for regression. Linear regression finds the parameters 
a and b that minimize the mean squared error between predictions and the true 
regression targets, y, on the training set.

For small datasets the linear regression model is very simple and is 
underfitted but for higher-dimensional datasets the linear model becomes
powerfull and have a higher chance of overfitting.
"""
# =============================================================================
# Import Libraries
# =============================================================================
import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

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
mglearn.plots.plot_linear_regression_wave()
#where a[0]: 0.393906  b: -0.031804

" Scikit-Learn "
#training and test data from wave dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))            # a parameter (slope)
print("lr.intercept_: {}".format(lr.intercept_))  # b parameter (intercept)
#lr.coef_: [ 0.394]
#lr.intercept_: -0.031804343026759746

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#Training set score: 0.67
#Test set score: 0.66

# =============================================================================
# Real-World Boston Dataset
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, random_state=0)

lr = LinearRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#Training set score: 0.95
#Test set score: 0.61
#the difference between the training and testing dataset is a clear sign of
#overfitting and we should therefore find a model which led us control the
#complexity - ridge regression

" Ridge Regression"