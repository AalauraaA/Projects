# -*- coding: utf-8 -*-
# =============================================================================
# Group ID: 234
# Members: Laura Nyrup, Trine Jensen, Christian Toft.
# Date: 2018-09-12
# Lecture 3 - Parametric and nonparametric methods
# Dependencies: numpy, matplotlib.pyplot, scipy.stats
# Python version: 3.6
# Functionality: Script that calculates the prior probability for a given point
# by a gausien model using parameter estimation from the training set.
# =============================================================================
"""
Exercise: 
Download dataset1_noisy (generated from handwritten digits database) 
available in the end of this page and do the following exercise:
    You are given, as the train data, trn_x and trn_y along with 
    their class labels trn_x_class and trn_y_class. The task is to classify 
    the following TEST data.
    (a) classify instances in tst_xy, and use the corresponding label file 
    tst_xy_class to calculate the accuracy;
    (b) classify instances in tst_xy_126 by assuming a uniform prior over the 
    space of hypotheses, and use the corresponding label file tst_xy_126_class 
    to calculate the accuracy;
    (c) classify instances in tst_xy_126 by assuming a prior probability of 
    0.9 for Class x and 0.1 for Class y, and use the corresponding label file 
    tst_xy_126_class to calculate the accuracy; compare the results with those 
    of (b).
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# =============================================================================
# Data import for the training set
# =============================================================================
trn_x = np.loadtxt("trn_x.txt")         # Training data import
trn_x_t = np.transpose(trn_x)
class_x = np.loadtxt("trn_x_class.txt") # Labels import

trn_y = np.loadtxt("trn_y.txt")         # Training data import
trn_y_t = np.transpose(trn_y)
class_y = np.loadtxt("trn_y_class.txt") # Labels import

" Plot of the training data "
plt.figure(1)
plt.scatter(trn_x_t[0], trn_x_t[1])
plt.show()

plt.figure(2)
plt.scatter(trn_y_t[0], trn_y_t[1])
plt.show()

# =============================================================================
# Parameter estimation
# =============================================================================
" Mean values "
mean_trn_x = np.mean(trn_x, axis=0)
mean_trn_y = np.mean(trn_y, axis=0)

" Covariance matrix "
cov_trn_x = np.cov(trn_x_t)
cov_trn_y = np.cov(trn_y_t)

" Prior probability based on the data amount (roughly 55/45) "
prior_x = len(trn_x)/(len(trn_x)+len(trn_y))
prior_y = len(trn_y)/(len(trn_x)+len(trn_y))

" Model for the multivariate normal with the given parameters "
model_x = ss.multivariate_normal(mean_trn_x, cov_trn_x)
model_y = ss.multivariate_normal(mean_trn_x, cov_trn_y)

def classification(mod_x, mod_y, prior_x, prior_y, data):
    """
    Using a calculated model for the multivariate normal distribution
    the maximum prosterior probability is calculated .

    The probability for the point belonging to the different classes is
    calculated. This to gether with the prior probability gives the
    posterior.

    If the posterior is larger for one class than the other the function
    assign the class to the given data point.
    """
    classes = np.zeros(len(data))
    for i in range(len(data)):
        evidence = 1  # mod_x.pdf(data[i])*prior_x + mod_y.pdf(data[i])*prior_y
        post_c1 = ((mod_x.pdf(data[i]))*prior_x)/(evidence)
        post_c2 = ((mod_y.pdf(data[i]))*prior_y)/(evidence)

        if post_c1 >= post_c2:  # Obs bias for 1 since it has higher prior
            classes[i] = 1
        elif post_c2 >= post_c1:
            classes[i] = 2
        else:
            classes[i] = -1     # Test to see if its doing something wrong.

    return classes

# =============================================================================
# Exercise a)
# =============================================================================
" Loading the test data set "
tst_xy = np.loadtxt("tst_xy.txt")
tst_xy_class = np.loadtxt("tst_xy_class.txt")

" Number of wrongly classified points "
resultat = classification(model_x, model_y, prior_x, prior_y, tst_xy)
# resultat = classification(mod_x, mod_y, 0.70, 0.30, tst_xy)

a_percentile = (np.count_nonzero(resultat-tst_xy_class)/len(resultat))
a_percentile = a_percentile *100
print('Test XY percentile difference: ', a_percentile)

correct = tst_xy[np.equal(tst_xy_class, resultat)]

" Plot of the test data for visualisation "
plt.figure(3)
plt.scatter(tst_xy.T[0], tst_xy.T[1], label='Estimated Classes')
plt.scatter(correct.T[0], correct.T[1], label='Correct Classes')
plt.title('Test Data XY')
plt.legend()
plt.show()

# =============================================================================
# Exercise b) with prior probability 50%
# =============================================================================
" Loading the test data set "
tst_xy_126 = np.loadtxt('tst_xy_126.txt')
tst_xy_126_class = np.loadtxt('tst_xy_126_class.txt')

" Prior probability uniform "
b_resultat = classification(model_x, model_y, 0.5, 0.5, tst_xy_126)

b_percentile = (np.count_nonzero(b_resultat-tst_xy_126_class)/len(resultat))
b_percentile = b_percentile * 100
print('Test XY_126 percentile difference: ', b_percentile)

# =============================================================================
# Exercise c) with prior probability 0.9 for X and 0.1 for Y
# =============================================================================
c_resultat = classification(model_x, model_y, 0.9, 0.1, tst_xy_126)

c_percentile = (np.count_nonzero(c_resultat-tst_xy_126_class)/len(resultat))
c_percentile = c_percentile*100
print('Test XY_126 percentile difference: ', c_percentile)
