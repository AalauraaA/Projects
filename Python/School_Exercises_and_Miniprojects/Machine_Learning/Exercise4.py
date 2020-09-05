# -*- coding: utf-8 -*-
# =============================================================================
# Group ID: 234
# Members: Laura Nyrup, Trine Jensen, Christian Toft.
# Date: 2018-10-10
# Lecture 3 Parametric and nonparametric methods
# Dependencies: numpy, scipy.stats, scipy.io, sklearn.decomposition,
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
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from baysean_classfier import baysean_cls_gauss as bcg

# =============================================================================
# Data import and variable
# =============================================================================
data = io.loadmat("Data/mnist_all.mat")  # Import
# Split picking out the relevant data set.
k = 10

def accuracy(cls, test_labels):
        """
        Performs classification and compares with the class labels assuming
        integer lables.
        """
        N = len(test_labels)

        # Calculate total correct as precentage
        total_correct = 100*(N - np.count_nonzero(cls - test_labels))/N

        # Calculate precentag correct for each class
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
# Data management
# =============================================================================
# label_dict
N_data = {i: len(data['train'+str(i)]) for i in range(k)}

# concatonated if needed (It is)
AVD = np.vstack((data['train'+str(i)] for i in range(k)))

# Arrayed DATA
AD = np.array([data['train'+str(i)] for i in range(k)])

# concatonated if needed (It is)
label = np.hstack((np.ones(len(data['train'+str(i)]))*i for i in range(k)))

# Test data
TD = np.vstack((data['test'+str(j)] for j in range(k)))

# List of labels
l_t = np.hstack((np.ones(len(data['test'+str(i)]))*i for i in range(k)))

# =============================================================================
# LDA. k = #class - 1
# =============================================================================
lda = LinearDiscriminantAnalysis(n_components=k-1)
pca = PCA(n_components=k-1)

x_r = lda.fit(AVD, label).transform(AVD)
x_r2 = pca.fit(AVD).transform(AVD)

TTD = pca.fit(AVD).transform(TD)
pre_lda = lda.predict(TD)

print("\n")
print("LDA with a dimensional reduction to 9")
tc, cc = accuracy(pre_lda, l_t)
print("\n")
print("Equivelant PCA dimensional reduction")
model = bcg(x_r2, label, N_data)
model.print_accuracy(TTD, l_t)
print("\n")

# =============================================================================
# LDA. k = #class - 8
# =============================================================================
lda = LinearDiscriminantAnalysis(n_components=k-8)
pca = PCA(n_components=k-8)

x_r = lda.fit(AVD, label).transform(AVD)
x_r2 = pca.fit(AVD).transform(AVD)

TTD = pca.fit(AVD).transform(TD)
pre_lda = lda.predict(TD)

print("\n")
print("LDA with a dimensional reduction to 2")
tc, cc = accuracy(pre_lda, l_t)
print("\n")
print("Equivelant PCA dimensional reduction")
model = bcg(x_r2, label, N_data)
model.print_accuracy(TTD, l_t)
print("\n")

# =============================================================================
# %% plot of the 2 dimmensional case for vissualisation
# =============================================================================
plt.figure()
lw = 2

for i in range(k):
    plt.scatter(x_r2[label == i, 0], x_r2[label == i, 1], alpha=.8, lw=lw,
                label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of MNIST dataset')

plt.figure()
for i in range(k):
    plt.scatter(x_r[label == i, 0], x_r[label == i, 1], alpha=.8,
                label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of MNIST dataset')

plt.show()
