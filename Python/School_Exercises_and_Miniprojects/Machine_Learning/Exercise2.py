# -*- coding: utf-8 -*-
# =============================================================================
# Group ID: 234
# Members: Laura Nyrup, Trine Jensen, Christian Toft.
# Date: 2018-09-27
# Lecture 4 - Dimensionality Reduction
# Dependencies: numpy, matplotlib.pyplot, scipy.stats, scipy.io
# Python version: 3.6
# Functionality: Dimensional reduction using PCA followed by a
# Script that calculates the prior probability for a given point
# by a gausien model using parameter estimation from the training set.
# =============================================================================
import numpy as np
import scipy.io as io
import scipy.stats as ss
import matplotlib.pyplot as plt

# =============================================================================
# Dimension reduction using PCA
# =============================================================================
data = io.loadmat("Data/mnist_all.mat")  # Import data set

# Train data stacked
UD = np.vstack((np.vstack((data['train5'], data['train6'])), data['train8']))

cov = np.cov(UD.T) # Covariance matrix is calculated

U, S, V = np.linalg.svd(cov) # PCA, using the SVD command

# Basis for dimension containing the most information
# That is the two eigenvectors corresponding to the highest eigenvalues
BasisV1 = V[0]
BasisV2 = V[1]

# The data projected onto the new found basis
trn_x5 = np.dot(data['train5'], BasisV1)
trn_y5 = np.dot(data['train5'], BasisV2)
trn_x6 = np.dot(data['train6'], BasisV1)
trn_y6 = np.dot(data['train6'], BasisV2)
trn_x8 = np.dot(data['train8'], BasisV1)
trn_y8 = np.dot(data['train8'], BasisV2)

xud = np.dot(UD, BasisV1)
yud = np.dot(UD, BasisV2)

# =============================================================================
# Optional plot for visualisation
# =============================================================================
plt.figure('The collective data set')
plt.scatter(xud, yud, label='Intire data')
plt.legend()

plt.figure('The data sets visualised with different colors')
plt.scatter(trn_x5, trn_y5, label='5')
plt.scatter(trn_x6, trn_y6, label='6')
plt.scatter(trn_x8, trn_y8, label='8')
plt.legend()

# =============================================================================
# Parameter estimation
# =============================================================================
# Mean values
mean_trn_5 = np.mean((trn_x5, trn_y5), axis=1)
mean_trn_6 = np.mean((trn_x6, trn_y6), axis=1)
mean_trn_8 = np.mean((trn_x8, trn_y8), axis=1)

# Covariance matrix
cov_trn_5 = np.cov((trn_x5, trn_y5))
cov_trn_6 = np.cov((trn_x6, trn_y6))
cov_trn_8 = np.cov((trn_x8, trn_y8))

# Prior probability based on the data amount
prior_5 = len(trn_x5)/(len(trn_x5)+len(trn_x6)+len(trn_x8))
prior_6 = len(trn_x6)/(len(trn_x5)+len(trn_x6)+len(trn_x8))
prior_8 = len(trn_x8)/(len(trn_x5)+len(trn_x6)+len(trn_x8))

# Model for the multivariate normal distribution with the given parameters
mod_5 = ss.multivariate_normal(mean_trn_5, cov_trn_5)
mod_6 = ss.multivariate_normal(mean_trn_6, cov_trn_6)
mod_8 = ss.multivariate_normal(mean_trn_8, cov_trn_8)

def classification(mod_x, mod_y, mod_z, prior_x, prior_y, prior_z, data):
    """
    Using a calculated model for the multivariate normal distribution
    the maximum prosterior probability is calculated.

    The probability for the point belonging to the different classes is
    calculated. This together with the prior probability gives the
    posterior.

    If the posterior is larger for one class than the other the function
    assign the class to the given data point.
    """
    klasse = np.zeros(len(data))
    for i in range(len(data)):
        Evidence = 1  # mod_x.pdf(data[i])*prior_x + mod_y.pdf(data[i])*prior_y
        post_c1 = ((mod_x.pdf(data[i]))*prior_x)/(Evidence)
        post_c2 = ((mod_y.pdf(data[i]))*prior_y)/(Evidence)
        post_c3 = ((mod_z.pdf(data[i]))*prior_z)/(Evidence)

        if post_c1 >= post_c2 and post_c1 >= post_c3:
            klasse[i] = 1
        elif post_c2 >= post_c1 and post_c2 >= post_c3:
            klasse[i] = 2
        elif post_c3 >= post_c2 and post_c3 >= post_c1:
            klasse[i] = 3
        else:
            klasse[i] =-1  # Test to see if its doing something wrong.
    return klasse


# =============================================================================
# Creating a test set
# =============================================================================
" Test for classificationg test5 "
testx = np.dot(data['test5'], BasisV1)
testy = np.dot(data['test5'], BasisV2)
test = np.vstack((testx, testy)).T

klass = classification(mod_5, mod_6, mod_8, prior_5, prior_6, prior_8, test)

c_per = np.count_nonzero(klass-1) # number of classifications that was not class 1(number 5) - misclassifications
c_per = c_per/len(klass)
print('Test5:', 1-c_per)

" Test for classificationg test6 "
testx = np.dot(data['test6'], BasisV1)
testy = np.dot(data['test6'], BasisV2)
test = np.vstack((testx, testy)).T

klass = classification(mod_5, mod_6, mod_8, prior_5, prior_6, prior_8, test)

c_per = np.count_nonzero(klass-2)
c_per = c_per/len(klass)
print('Test6:', 1-c_per)

" Test for classificationg test8 "
testx = np.dot(data['test8'], BasisV1)
testy = np.dot(data['test8'], BasisV2)
test = np.vstack((testx, testy)).T

klass = classification(mod_5, mod_6, mod_8, prior_5, prior_6, prior_8, test)

c_per = np.count_nonzero(klass-3)
c_per = c_per/len(klass)
print('Test8:', 1-c_per)
