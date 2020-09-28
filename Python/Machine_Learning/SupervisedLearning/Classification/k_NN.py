# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 14:12:06 2020

@author: Laura
------------------------------------------------------------------------------
SUPERVISED LEARNING - CLASSIFICATION
------------------------------------------------------------------------------
Supervised learning is to build a model on training data and then be able to 
make accurate prediction on new, unseen data with the same characteristics as
the training data. If it make accurate prediction the model is generalize, and
we want to have a model to generalize as accurately as possible.
Building a model to complex lead to the model being overfitted but building 
the model to simple lead to the model being underfitted.
------------------------------------------------------------------------------

------------------------------------------------------------------------------
k-Nearest Neighbour (k-NN) for Classification
------------------------------------------------------------------------------
k-NN is a simple ML algorithm as only consist of storing the training dataset.
k-NN then make a prediction for a new data point by finding the closest data 
points in the training data set (the nearest neighbour).

In its simple version the k-NN algorithm only consider one nearest neighbour, 
the closest training point. The prediction is then the known output for the 
closest training point.

When considering k nearest neighbours (more than one), we then use a voting to 
assign a class label. That means that for each test point, we count how many 
neighbours belong to class 0 and how many belong to class 1. The most frequent 
class is then assign to the prediction point.

For a two-dimensional dataset we can also illustrate the predictions for all
possible test points in the xy-plane by a decision boundary. A smoother 
boundary corresponds to a simpler model.

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
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# Data Import
# =============================================================================
" Synthetic Forge Data Set "
X, y = mglearn.datasets.make_forge() # generate dataset
print("X.shape: {}".format(X.shape))

# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

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
# Synthetic Forge Dataset
# =============================================================================
" k = 1 "
mglearn.plots.plot_knn_classification(n_neighbors=1)

" k = 3 "
mglearn.plots.plot_knn_classification(n_neighbors=3)

" Scikit-learn "
#training and test data from forge dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3) # k-NN with k = 3
clf.fit(X_train, y_train) # fit the classifier (storing the dataset)

#for each test data point in the test dataset its compute the nearest neighbour
print("Test set predictions:\n{}".format(clf.predict(X_test)))
#Test set predictions: [1 0 1 0 1 0 0]

#to evaluate how well the model generalize, we use the score method
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
#Test set accuracy: 0.86
#the k-NN model is 86% accurate meaning that the model predict the class label
#correctly for 86% of the samples in the test dataset.

" Decision Boundary - k = 1, 3 and 9 "
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbours, ax in zip([1, 3, 9], axes):
    # the fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbours).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbour(s)".format(n_neighbours))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)

# =============================================================================
# Real-World Breast Cancer Dataset
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = [] # save the accuracy
test_accuracy = []     # save the accuracy

neighbours_settings = range(1, 11) # try n_neighbors from 1 to 10

for n_neighbours in neighbours_settings:
    #build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbours)
    clf.fit(X_train, y_train)
    
    #record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    #record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
    
plt.plot(neighbours_settings, training_accuracy, label="training accuracy")
plt.plot(neighbours_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbours")
plt.legend()

"""
The training dataset perform well with k = 1 but is dropping as k increase. 
The test dataset is varying increasing and decreasing but has it best 
performance for k = 6. The more neighbours (k) used the simple the model 
become and there for a lower accuracy. Therefore it is a trade-off leading to
k = 6 being the best choice. After all the lowest accuracy is 88 % which is
high.
"""