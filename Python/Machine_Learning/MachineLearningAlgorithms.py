# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 09:34:01 2020

@author: Laura


Machine Learning Algorithms
"""
import pandas as pd
from sklearn import linear_model, tree, svm, decomposition
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv('iris.csv')

data = dataset.drop('Id', axis=1)
x = data.drop('Species', axis=1) # Values
y = data['Species']              # Labels

" 80% training data and 20% validation data "
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# =============================================================================
# Linear Regression
# =============================================================================
linear = linear_model.LinearRegression()

# Train the model using training data
linear.fit(x_train, y_train)

# Check the score
linear.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

# Predict output
predicted = linear.predict(x_test)

# =============================================================================
# Logistic Regression
# =============================================================================
model = LogisticRegression()

# Train the model using training data
model.fit(x_train, y_train)

# Check the score
model.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict output
predicted = model.predict(x_test)

# =============================================================================
# Decision Tree
# =============================================================================
model_class = tree.DecisionTreeClassifier(criterion='gini') # for classification
model_reg = tree.DecisionTreeRegressor()                    # for regression

# Train the model using training data
model_class.fit(x_train, y_train)
model_reg.fit(x_train, y_train)

# Check the score
model_class.score(x_train, y_train)
model_reg.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model_class.coef_)
print('Intercept: \n', model_class.intercept_)

print('Coefficient: \n', model_reg.coef_)
print('Intercept: \n', model_reg.intercept_)

# Predict output
predicted = model_class.predict(x_test)
predicted = model_reg.predict(x_test)

# =============================================================================
# Support Vector Machine (SVM)
# =============================================================================
model = svm.svc()

# Train the model using training data
model.fit(x_train, y_train)

# Check the score
model.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict output
predicted = model.predict(x_test)

# =============================================================================
# Naive Bayes
# =============================================================================
model = GaussianNB()

# Train the model using training data
model.fit(x_train, y_train)

# Check the score
model.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict output
predicted = model.predict(x_test)

# =============================================================================
# K-Nearest Neighbors (kNN)
# =============================================================================
model = KNeighborsClassifier(n_neighbors=6) #default is 5
# Train the model using training data
model.fit(x_train, y_train)

# Check the score
model.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict output
predicted = model.predict(x_test)

# =============================================================================
# k-Means
# =============================================================================
model = KMeans(n_clusters=3, random_state=0)

# Train the model using training data
model.fit(x_train, y_train)

# Check the score
model.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict output
predicted = model.predict(x_test)

# =============================================================================
# Random Forest
# =============================================================================
model = RandomForestClassifier()

# Train the model using training data
model.fit(x_train, y_train)

# Check the score
model.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict output
predicted = model.predict(x_test)

# =============================================================================
# Dimensionality Reduction Algorithms 
# =============================================================================
pca = decomposition.PCA()
fa = decomposition.FactorAnalysis()

# Train the model using training data
train_reduced = pca.fit_transform(x_train)
test_reduced = pca.transform(x_test)

# =============================================================================
# Gradient Boosting & AdaBoost
# =============================================================================
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Train the model using training data
model.fit(x_train, y_train)

# Check the score
model.score(x_train, y_train)

# Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

# Predict output
predicted = model.predict(x_test)









