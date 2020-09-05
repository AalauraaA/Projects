# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:06:42 2018

@author: Laura
"""
import numpy as np
import scipy.io as io

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

# =============================================================================
# %% Data import
# =============================================================================
data = io.loadmat("mnist_all.mat")

k=10

x_train = np.vstack((data['train'+str(j)] for j in range(k)))
y_train = np.hstack((np.ones(len(data['train'+str(i)]))*i for i in range(k)))

x_test = np.vstack((data['test'+str(j)] for j in range(k)))
y_test = np.hstack((np.ones(len(data['test'+str(i)]))*i for i in range(k)))

# =============================================================================
# %% AdaBoost
# =============================================================================
bdt_real = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=50,
    learning_rate=1)

bdt_real.fit(x_train, y_train)

# =============================================================================
# %% Error calculating
# =============================================================================
real_test_errors = [] # Accracy score between predicted label and real label (test data)

for real_test_predict in bdt_real.staged_predict(x_test):
    real_test_errors.append(1. - accuracy_score(real_test_predict, y_test))

n_trees_real = len(bdt_real) # 50 trees

real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
# =============================================================================
# %% Plots
# =============================================================================
plt.figure(1)
plt.plot(range(1, n_trees_real + 1),
         real_test_errors, c='black',
         linestyle='dashed', label='Real AdaBoost')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')
plt.title('Accuracy between predict and real labels')

plt.figure(2)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
         "r", label='Real AdaBoost', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.title('Classification error for each estimator')
plt.show()
      