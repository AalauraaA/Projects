# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:40:33 2020

@author: Laura

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
Linear Models for Classification
------------------------------------------------------------------------------
Linear models make a prediction using a linear function of the input features. 
This is for a binary classification:
    y = a[0] * x[0] + a[1]*x[1] + ... + a[p] * x[p] + b < 0
Here x[0]-x[p] is the features of a single data point, a and b are parameters
of the model that is learned. If the linear function is smaller than 0 then
it is classified as -1 and if it is larger than 0 then it is classified as +1 

Linear models for classification the decision boundary is a linear function of
the input. A linear classifier is a classifier that separates two classes 
using a line, a plane, or a hyperplane.

The most common linear classification algorithms are the logistic regression
and linear support vector machine (SVM). Both methods used the L2 
regularization as default. To achieve better result one should perform a
trade-off with parameter C that determine the strenght of the regularization.
    * A high C lead to less regularization and therefore the algorithms try to 
      fit the training dataset as best as possible. 
      
      High C will stresses the importance that each individual data point
      be classified correctly.
      
    * A low C lead to a high regularization and therefore the algorithms try 
      to put more emphasis on finding a cofficient vector 'a' that is close 
      to zero.
      
      Low C will also cause the algorithms to try to adjust to the 'majority'
      of the data points.
"""
# =============================================================================
# Import Libraries
# =============================================================================
import mglearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# =============================================================================
# Data Import
# =============================================================================
" Synthetic Forge Data Set "
X, y = mglearn.datasets.make_forge() # generate dataset
print("X.shape: {}".format(X.shape))

# plot dataset
plt.figure(1)
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
# Synthetic Forge Dataset - Binary Classification - Logistic Regression
# =============================================================================
" Scikit-learn "
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

" Trade-off C Parameter "
mglearn.plots.plot_linear_svc_regularization()

"""
A very small C corresponding to a lot of regularization. The strongly 
regularized model chooses a relatively horizontal line, misclassifying
two points. 

A slightly higher C lead to the model being focuses more on the two 
misclassified samples, tilting the decision boundary.

A very high value of C the model tilts the decision boundary a lot, now
correctly classifying all points in class 0. One of the points in class 1 is 
still misclassified. In other words, this model is likely overfitting.
"""

# =============================================================================
# Real-World Breast Cancer Dataset - Binary Classification - Logistic Regression
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)

" C = 1 "
logreg = LogisticRegression().fit(X_train, y_train)

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
#Training set score: 0.953
#Test set score: 0.958

" C = 100 "
logreg100 = LogisticRegression(C=100).fit(X_train, y_train)

print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))
#Training set score: 0.972
#Test set score: 0.965

" C = 0.01 "
logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train) # C = 0.01

print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))
#Training set score: 0.934
#Test set score: 0.930

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()

#limit the model to few feature for better interpretabling - L1 regularization
for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_test, y_test)))

#Training accuracy of l1 logreg with C=0.001: 0.91
#Test accuracy of l1 logreg with C=0.001: 0.92
#Training accuracy of l1 logreg with C=1.000: 0.96
#Test accuracy of l1 logreg with C=1.000: 0.96
#Training accuracy of l1 logreg with C=100.000: 0.99
#Test accuracy of l1 logreg with C=100.000: 0.98
    
plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.ylim(-5, 5)
plt.legend(loc=3)











