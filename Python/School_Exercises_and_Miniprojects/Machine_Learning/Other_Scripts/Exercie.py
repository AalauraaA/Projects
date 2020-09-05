# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 10:17:49 2018

@author: laura

Perform classification for the entire MNIST dataset based on the algorithms 
introduced: 
    Use LDA for dimensionality reduction to 2 or 9 dimensions, 
    classify the dimension-reduced data
    compare this classification performance with that of using PCA. 
"""
import numpy as np
import scipy.io as io
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data = io.loadmat("mnist_all.mat")  # Import


X = np.vstack((np.vstack((data['train5'], data['train6'])), data['train8']))
y = np.array([np.ones([len(data['train5']),1])*5, 
              np.ones([len(data['train6']),1])*6, 
              np.ones([len(data['train8']),1])*8 ])
y = np.vstack((y[i] for i in range(len(y))))

lda = LinearDiscriminantAnalysis(n_components=2, priors=None, shrinkage=None,
              solver='svd', store_covariance=False, tol=0.0001)

X_dim2 = lda.fit(X, y).transform(X)

target_names = np.array(["5", "6", "8"])

plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0,1,2], target_names):
    plt.scatter(X_dim2[y == i, 0], X_dim2[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')