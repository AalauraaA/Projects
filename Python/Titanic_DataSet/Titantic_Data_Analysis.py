# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:17:26 2020

@author: Laura

Data Analysis of the Titanic Data Set
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# Import of the data
# =============================================================================
titanic = pd.read_csv('titanic.csv')
print(titanic.head())
print("")

# =============================================================================
# Clean the data
# =============================================================================
print(titanic.isnull().sum()) # looks for missing values
print("")

# We fill in the missing values in Age with the mean of all existing Age values 
# because the Age is missing at random.
titanic['Age'].fillna(titanic['Age'].mean(), inplace=True)
print(titanic.isnull().sum())

# Lets look at how many there survived on Titanic. Here we combine the SibSp, Pclass and Parch and made a 
# categorical plot.
for i, col in enumerate(['Pclass', 'SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2,)
# From the plots we can see that as people have more siblings or spouses aboard, they're less likely to 
# survive. Beacuse the two features are so closely related we want to combine them into one feature and 
# by that clean the model even more.

# Lets remove all the unneccesary data which is not need for this analysis. the Parch. Drop all categorical 
# features
cat_feat = ['PassengerId', 'Pclass', 'SibSp', 'Parch', 'Name', 'Ticket', 'Sex', 'Cabin', 'Embarked']
titanic.drop(cat_feat, axis=1, inplace=True)
print(titanic.head())
print("")

print(titanic.describe()) # Explore Continuos Features
print("")

# From the above table we can see that there are some missing values in the Age 
# columns as there only is count 714 but there is 891 passenger. Another thing 
# is that the Survived column, the target variable, is binary and therefore 
# only have the value 0 or 1. Because it is binary we can use the mean to 
# tell which percentage of the people survived in this data set.
print(titanic.groupby('Survived').mean())
print("")

print(titanic.groupby(titanic['Age'].isnull()).mean()) # Look for missing values in Age are random or not
print("")

for i in ['Age', 'Fare']:
    died = list(titanic[titanic['Survived'] == 0][i].dropna())
    survived = list(titanic[titanic['Survived'] == 1][i].dropna())
    xmin = min(min(died), min(survived))
    xmax = max(max(died), max(survived))
    width = (xmax - xmin) / 40
    sns.distplot(died, color='r', kde=False, bins=np.arange(xmin, xmax, width))
    sns.distplot(survived, color='g', kde=False, bins=np.arange(xmin, xmax, width))
    plt.legend(['Did not survive', 'Survived'])
    plt.title('Overlaid histogram for {}'.format(i))
    plt.show()



titanic['family_cnt'] = titanic['SibSp'] + titanic['Parch']
sns.catplot(x='family_cnt', y='Survived', data=titanic, kind='point', aspect=2,)







