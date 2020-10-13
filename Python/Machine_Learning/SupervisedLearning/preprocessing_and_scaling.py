# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:04:29 2020

@author: Laura
------------------------------------------------------------------------------
UNSUPERVISED LEARNING - UNSUPERVISED TRANSFORMATION
------------------------------------------------------------------------------
There is two types of unsupervised learning: unsupervised transformation and 
clustering.

With unsupervised transformation a dataset is transform into a new representation
of the dataset with some algorithms such as the dimensionality reduction or 
extraction of features/topics.

With clustering the dataset is partition into distinct groups of similar items.

The major challenge with unsupervised learning is that you do not have a known 
output to evaluate the algorithm to see if it good. You use unsupervised 
learning algorithms when your data do not contain any labels. Because of this 
unsupervised learning algorithms are often used in a exploratory setting to 
understand the data better than used it as an automatic system. Unsupervised
learning is also used as a preprocessing step for the supervised algorithms as
the knowlegde of the data can help the accuracy of the supervised algorithms.
------------------------------------------------------------------------------

------------------------------------------------------------------------------
Preprocessing and Scaling Functions
------------------------------------------------------------------------------
The following four functions are different ways to transform the data and by
that make some preprocessing:
    * StandardScaler: This transformation function ensure that for each feature
      the mean is 0 and variance is 1, giving all the feature the same magintude.
      It does not ensure minimum/maximum values for the features.
      
    * RobustScaler: This transformation work similar to the StandardScaler but
      it use the median and quartiles instead of mean and variance. That make 
      it ignore data point which differ from the rest (the outliers)
      
    * MinMaxScaler: This transformation function shifts the data such that all
      the features lays between 0 and 1.
      
    * Normalizer: This transformation function scales each data point such that
      the feature vector has a Euclidean lenght of 1. It projects a data point
      on the circle with a radius 1.

A scaling method is usually applied before applying a supervised ML algorithm.
"""
# =============================================================================
# Libraries Import
# =============================================================================
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =============================================================================
# Data Import
# =============================================================================
" Real World Classification Data Set - Breast Cancer "
#task is to learn to predict whether a tumor is maligant based on the 
#measurements of the tissue
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

print("Shape of cancer data: {}".format(cancer.data.shape))
#contain 569 data points and 30 features

print("Sample counts per class:\n{}".format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
#out of 569 data points, 212 are maligant and 357 are benign

print("Feature names:\n{}".format(cancer.feature_names))

# =============================================================================
# MinMaxScaler Scaling + SVM - Breast Cancer Dataset 
# =============================================================================
" Train and test data "
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)
print("Shape of the cancer training data: {}".format(X_train.shape))
#shape = (426, 30)
print("")
print("Shape of the cancer test data: {}".format(X_test.shape))
#shape = (143, 30)

" MinMaxScaler Scaling Function "
scaler = MinMaxScaler()
scaler.fit(X_train)

" Transform data "
X_train_scaled = scaler.transform(X_train)

# print dataset properties before and after scaling
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
X_train_scaled.max(axis=0)))

" Transform test data "
X_test_scaled = scaler.transform(X_test)

# print test data properties after scaling
print("per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
#because we use the training data fit function 'scale' to transform our test data
#the test date will be scale according to the minimum/maximum values of the 
#training dataset and therefore the transform test dataset do not contain 
#perfect 0 and 1 and some point are even not in the interval.
#it is important to used the same transformation method on both dataset to
#make the learning correctly.

" SVM on unscale data "
svm = SVC(C=100)
svm.fit(X_train, y_train)
# scoring on the unscaled test set
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
#Test set accuracy: 0.63

" SVM on scale data "
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("Scaled test set accuracy: {:.2f}".format(
svm.score(X_test_scaled, y_test)))
#Scaled test set accuracy: 0.97

# =============================================================================
# StandardScaler Scaling + SVM - Breast Cancer Dataset 
# =============================================================================
" Train and test data "
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
random_state=1)
print("Shape of the cancer training data: {}".format(X_train.shape))
#shape = (426, 30)
print("")
print("Shape of the cancer test data: {}".format(X_test.shape))
#shape = (143, 30)

" StandardScaler Scaling Function "
scaler = StandardScaler()
scaler.fit(X_train)

" Transform data "
X_train_scaled = scaler.transform(X_train)

" Transform test data "
X_test_scaled = scaler.transform(X_test)

" SVM on scale data "
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("SVM test accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
#SVM test accuracy: 0.96

# =============================================================================
# StandardScaler + Shortcut - Cancer Breat Dataset
# =============================================================================
" Scaling Function "
scaler = StandardScaler()

# calling fit and transform in sequence (using method chaining)
X_scaled = scaler.fit(X_train).transform(X_train)

# same result, but more efficient computation
X_scaled_d = scaler.fit_transform(X_train)


