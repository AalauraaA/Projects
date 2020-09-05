# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 09:55:31 2018

@author: Not Mine but it is need for some of
         the execise scripts
"""

import numpy as np
from scipy.stats import multivariate_normal

"""
N_data should be something like
>>> N_data = {0: 10, 1: 20, 2: 4}
If you have classes 0, 1 and 2 with respectively 10, 20 and 4 observations

for each class.
Get the class by calling it like:
>>> models =  baysean_cls_gauss(train_data, train_lables, N_data)

Now you can get accuracy from test_data and its lables
>>> models.print_accuracy(test_data, test_lables)

And it will print out
"""


class baysean_cls_gauss:
    """
    Fit Gaussian modles to labled training data and can later perform
    classification on test data

    Parameters (for init)
    ----------

    X : np.array of dimention N x D
        Datapoints in rows each of dimention D
    y : np.array of dimention N as int-type
        Labels for datapoints in list as integers
    N_data : dict
        Dictionary with number of observations for each class (ordered like X)

    """
    def __init__(self, X, y, N_data=None):
        if N_data is None:
            raise(NotImplementedError("Sorry... not implemented yet"))

        # Split data into each labled class
        N_cumsum = np.cumsum(list(N_data.values()))
        X_split = np.split(X, N_cumsum)

        # Create appropiate dictionaries
        self.lab = np.unique(y)
        X_lab = {self.lab[i]: X_split[i] for i in range(len(self.lab))}
        self.models = dict.fromkeys(self.lab)
        for key in self.models.keys():
            self.models[key] = self.fit_gauss(X_lab[key])

        # Total number of datapoints
        N = sum(N_data.values())
        self.priors = {label: N_data[label]/N for label in self.lab}

    def fit_gauss(self, X):
        """
        Fit gaussian model for data matrix
        """
        mu = np.mean(X, axis=0)
        cov = np.cov(X.T)
        mod = multivariate_normal(mu, cov)
        return(mod)

    def classify(self, test_data, eps=1e-10):
        """
        Classifies data using the fitted modles.
        """

        # Calculate P(x|C_i) for all different classes
        PxC = {label: self.models[label].pdf(test_data) for label in self.lab}

        # Now calculate unnormalised P(C_i|x)
        PCx = {label: np.log(PxC[label] + eps) + np.log(self.priors[label])
               for label in self.lab}

        # Perform the classification
        arg_cls = np.argmax(np.vstack(PCx.values()), axis=0)
        cls = self.lab[arg_cls]

        return(cls)

    def accuracy(self, test_data, test_labels):
        """
        Performs classification and compares with the class labels assuming
        integer lables.
        """
        N = len(test_labels)
        cls = self.classify(test_data)

        # Calculate total correct as precentage
        total_correct = 100*(N - np.count_nonzero(cls - test_labels))/N

        # Calculate precentag correct for each class
        cls_correct = {}
        for label in self.lab:
            idx = np.where(test_labels == label)[0]
            N_cls = len(idx)
            cls_correct[label] = 100*(N_cls - np.count_nonzero(label -
                                                               cls[idx]))/N_cls

        return(total_correct, cls_correct)

    def print_accuracy(self, test_data, test_lables):
        """
        Prints out accuracies
        """

        total_correct, cls_correct = self.accuracy(test_data, test_lables)
        print("Accuracy for:")
        print("All classes is %.2f%%" % total_correct)
        for label in self.lab:
            print("Class %d is %.2f%%" % (label, cls_correct[label]))
        return(total_correct, cls_correct)
