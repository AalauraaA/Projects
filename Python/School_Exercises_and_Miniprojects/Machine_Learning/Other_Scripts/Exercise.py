# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 10:12:24 2018

@author: laura

This assignment is based on the previously generated 2-dimensional data of the 
three classes (5, 6 and 8) from the MNIST database of handwritten digits.
    a) First, mix the 2-dimensional data (training data only) by removing the 
       labels and then use one Gaussian mixture model to model them.
       
    b) Secondly, compare the Gaussian mixture model with the Gaussian models 
       trained in the previous assignment, in terms of mean and variance values 
       as well as through visualisation.
"""
import numpy as np
import scipy.io as io
import scipy.stats as ss
import matplotlib.pyplot as plt

data = io.loadmat("2D3classes.mat")  # PCA data to comparing 

trn_5 = data['trn5']
trn_6 = data['trn6']
trn_8 = data['trn8']

group_data = np.vstack([trn_5, trn_6, trn_8])

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
    prior = 0
    for i in range(len(data)):
        for j in range(prior):
            post_c1 = ((mod_x.pdf(data[i]))*prior_x)/((mod_x.pdf(data[i]))*prior_x[j])
            post_c2 = ((mod_y.pdf(data[i]))*prior_y)/((mod_y.pdf(data[i]))*prior_y[j])
            post_c3 = ((mod_z.pdf(data[i]))*prior_z)/((mod_z.pdf(data[i]))*prior_z[j])
    
        if post_c1 >= post_c2 and post_c1 >= post_c3:
            klasse[i] = 1
        elif post_c2 >= post_c1 and post_c2 >= post_c3:
            klasse[i] = 2
        elif post_c3 >= post_c2 and post_c3 >= post_c1:
            klasse[i] = 3
        else:
            klasse[i] = -1  # Test to see if its doing something wrong.
    return klasse


