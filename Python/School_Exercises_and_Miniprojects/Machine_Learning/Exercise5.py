# -*- coding: utf-8 -*-
# =============================================================================
# Group ID: 234
# Members: Laura Nyrup, Trine Jensen, Christian Toft.
# Date: 2018-10-10
# Lecture 3 Parametric and nonparametric methods
# Dependencies: numpy, scipy.stats, scipy.io, sklearn,
# sklearn.discriminant_analysis, matplotlib.pyplot
# Python version: 3.6
# Functionality: Dimensional reduction using LDA followed by a reduction using
# PCA.
# A Script that calculates the prior probability for a given point
# by a gausien model using parameter estimation from the training set.
# is further used for comparison between the LDA and PCA.
# =============================================================================
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

def accuracy(cls, test_labels):
        """
        Performs classification and compares with the class labels assuming
        integer lables.
        """
        N = len(test_labels)

        # Calculate total correct as percentage
        total_correct = 100*(N - np.count_nonzero(cls - test_labels))/N

        # Calculate percentage correct for each class
        lab = np.unique(test_labels)
        cls_correct = {}
        for label in lab:
            idx = np.where(test_labels == label)[0]
            N_cls = len(idx)
            cls_correct[label] = 100*(N_cls - np.count_nonzero(label -
                                                               cls[idx]))/N_cls

        print("Accuracy for:")
        print("All classes is %.2f%%" % total_correct)
        
        for label in lab:
            print("Class %d is %.2f%%" % (label, cls_correct[label]))
        
        return(total_correct, cls_correct)

# =============================================================================
# Data import and variable
# =============================================================================
data = io.loadmat("Data/mnist_all.mat")  # Import

# Split picking out the relevant data set.
k = 10
dr = 8

# =============================================================================
# Data management
# =============================================================================
# label_dict
N_data = {i: len(data['train'+str(i)]) for i in range(k)}

# concatonated if needed (It is)
AVD = np.vstack((data['train'+str(i)] for i in range(k)))

# Arrayed data
AD = np.array([data['train'+str(i)] for i in range(k)])

# concatonated if needed (It is)
label = np.hstack((np.ones(len(data['train'+str(i)]))*i for i in range(k)))

# Test data
TD = np.vstack((data['test'+str(j)] for j in range(k)))

# List of labels
l_t = np.hstack((np.ones(len(data['test'+str(i)]))*i for i in range(k)))

# =============================================================================
# Dimensional reduction to make it faster (LDA, k = 10-1)
# =============================================================================
lda = LinearDiscriminantAnalysis(n_components=k-dr)
x_r = lda.fit(AVD, label).transform(AVD)
TTD = lda.fit(AVD, label).transform(TD)

# =============================================================================
# Model fitting on the reduced data (kernel=linear)
# =============================================================================
model = svm.SVC(kernel='linear')

print('model')
model = model.fit(x_r, label)
predicted = model.predict(TTD)
print("\n")
print("LDA with a dimensional reduction to 9")
tc, cc = accuracy(predicted, l_t)
print("\n")

# =============================================================================
# Model fitting on the reduced data (kernel=rbf)
# =============================================================================
C = 5.0  # SVM regularization parameter (C=5, gamma=0.05)
model = svm.SVC(kernel='rbf', gamma=0.05, C=C)

print('model')
model = model.fit(x_r, label)
predicted = model.predict(TTD)
print("\n")
print("LDA with a dimensional reduction to 9")
tc, cc = accuracy(predicted, l_t)
print("\n")
