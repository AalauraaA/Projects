# -*- coding: utf-8 -*-
"""
Created on Sat May  9 16:38:10 2020

@author: Laura
------------------------------------------------------------------------------
Iris Flower Data Set
------------------------------------------------------------------------------
The Iris Flower Dataset or Fisher's Iris data set is a multivariate data set 
introduced by the British statistician and biologist Ronald Fisher in his 1936 
paper The use of multiple measurements in taxonomic problems as an example of 
linear discriminant analysis.

The data set consists of 50 samples from each of three species of Iris 
(Iris setosa, Iris virginica and Iris versicolor). Four features were measured 
from each sample: the length and the width of the sepals and petals, 
in centimeters. Based on the combination of these four features, Fisher 
developed a linear discriminant model to distinguish the species from each 
other.
------------------------------------------------------------------------------
url: https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
------------------------------------------------------------------------------
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# The ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# =============================================================================
# Load/Import Iris Flower Data Set
# =============================================================================
# Load dataset
dataset = pd.read_csv('iris.csv')

" Summarize the Data Set "
print('Shape of Iris Data Set: ', dataset.shape)
print("")
print('First 5 entries of Iris Data Set: \n ', dataset.head(5))
print("")
print(dataset.info())
print("")
print('The statistical summary of Iris Data Set: \n ', dataset.describe())
#.describe() prints the count, mean, min, max and percentiles of dataframe

print('The number of instance for each class/species of the Iris Data Set: ', dataset['Species'].value_counts())
print("")
print("")
# There is 50 instance for each classes/'species' (3 classes) leading to 33 %

" Looks for Interaction Between the Variables "
data = dataset.drop('Id', axis=1)
pd.plotting.scatter_matrix(data, marker='+', )
plt.show()
# Because of the diagonal grouping this suggest high correlation and 
# predictable relationship.

# ============================================================================
# Creating Training and Testing Data Sets
# ============================================================================
x = data.drop('Species', axis=1) # Values
y = data['Species']              # Labels

" 80% training data and 20% validation data "
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

"""
6 models will be tested:
    Logistic Regression (LR)
    Linear Discriminant Analysis (LDA)
    K-Nearest Neighbors (KNN).
    Classification and Regression Trees (CART).
    Gaussian Naive Bayes (NB).
    Support Vector Machines (SVM).
which both linear and nonlinear models.
"""

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

" Evaluate Each Model in Turn "
results = []
names = []
for name, model in models:
    # Used 10-fold cross-validation to estimate model accuracy
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('{}: CV (mean) {} CV (std) {}'.format(name, cv_results.mean(), cv_results.std()))

"""
LR: CV (mean) 0.9416666666666667 CV (std) 0.06508541396588878
LDA: CV (mean) 0.975 CV (std) 0.03818813079129868
KNN: CV (mean) 0.9583333333333333 CV (std) 0.04166666666666669
CART: CV (mean) 0.9499999999999998 CV (std) 0.04082482904638632
NB: CV (mean) 0.95 CV (std) 0.05527707983925667
SVM: CV (mean) 0.9833333333333332 CV (std) 0.03333333333333335

From the Cross-Validation results it can be seen that SVM has the largest 
estimated accuracy score which lays about 0.98 or 98%.
"""

" Model Evaluation Result Comparison "
plt.boxplot(results, labels=names)
plt.title('Model Evaluation Result Comparison')
plt.show()

# ============================================================================
# Prediction with SVM Model on Test Data Set
# ============================================================================
" Used the SVM Model to Predict with the Test Data "
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_test)

" Evaluate the Prediction "
print('Accuracy Score: {}'.format(accuracy_score(y_test, predictions)))
# Accuracy is 0.966 or about 96%
