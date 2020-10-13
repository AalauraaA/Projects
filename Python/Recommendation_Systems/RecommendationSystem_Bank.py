# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:38:11 2020

@author: Laura
------------------------------------------------------------------------------
Recommendation System in Python
------------------------------------------------------------------------------
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# =============================================================================
# Recommendation System Based on Machine Learning
# =============================================================================
" Logistic Regression as Classifier "
bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
print(bank_full.head())
print("")

# Lets view some information about the dataset
print(bank_full.info()) 
print("")

# Lets look at some of the data
X = bank_full.iloc[:,[18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]].values
y = bank_full.iloc[:,17].values

# Lets used the logistic regression model on the data
LogReg = LogisticRegression()
LogReg.fit(X, y) # fit the extracted data to the logistic regression model

# Let try to see if I can make a recommendation to a new user based on the 
# logistic regression model based on the selected data
new_user = np.matrix([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
new_user = new_user[0]
y_pred = LogReg.predict(new_user) # make some recommendation to the new user
print(y_pred)
print("")

print("Test set score: {:.3f}".format(LogReg.score(new_user, y_pred)))
print("")
#Test set score: 0.958

print("Training set score: {:.3f}".format(LogReg.score(X, y)))
print("")
#Training set score: 0.893

# print(classification_report(y, y_pred))
