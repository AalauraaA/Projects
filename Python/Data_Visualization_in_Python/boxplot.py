# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:27:50 2019

@author: Laura
@Url: https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51

Theme: Understanding Boxplots
-------------------------------------------------------------------------------
Boxplot is a way of displaying the distribution of data:
    * Minimum
    * First quartile (Q1)/25th percentile
        - The middel number between the smallest number which is not the minimum 
          and the median of the dataset.
    * Median (Q2)/50th percentile
        - The middel value of the dataset.
    * Third quartile (Q3)/75th percentile
        - The middel value between the median and the highest value which is 
          not the maximum of the dataset.
    - Maximum
    
A boxplot tells you about:
    * Outliers and their values
    * Symmetry in your data
    * How tightly your data is grouped
    * If your data is skewed

A boxplot is a graph that gives a good indication of how the values in the 
data are spread out. It take less space than a histogram or density plot
which come in hand when you want to comparing distributions between different
group of data.
"""
# =============================================================================
# Import Necessary Libraries
# =============================================================================
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Boxplot on a Normal Distribution
# =============================================================================
"""
The PDF is used to specify the probability of the random variable falling 
within a particular range of values. The probability is given by the integral
of this variables PDF over that range. 

The PDF for the normal distribution is given by:

     f(x) = 1.0 / np.sqrt(2*np.pi) * np.exp((-(x-mu)**2) / (2.0 * sigma2))

If we have mean/mu = 0 and the standard deviation/sigma2 = 1 then the PDF of
the normal distribution becomes a Gaussian distribution and is given as:
    
    f(x) = 1.0 / np.sqrt(2*np.pi) * np.exp((-x**2) / 2.0)

The area under the curve must be 1 (the probability of drawing any number from 
the function’s range is always 1).
"""
def normalProbabilityDensity(x):
    pdf = 1.0 / np.sqrt(2*np.pi) * np.exp((-x**2) / 2.0)
    return pdf
    
x = np.linspace(-4, 4, num = 100)
normal_pdf = normalProbabilityDensity(x)

# Visualization of Normal Distribution
plt.figure(1)
plt.plot(x, normal_pdf)
plt.ylim(0)
plt.title('Normal Distribution', size = 20)
plt.ylabel('Probability Density', size = 20)

"""
To get the probability of an event the pdf needs to be integrated. We want to 
find the probability of a random data point landing within the interquartile 
range 0.6745 standard deviation of the mean.
"""
# Integrate PDF from -.6745 to .6745
result_50p, _ = quad(normalProbabilityDensity, 
                     -0.6745, 
                     0.6745, 
                     limit = 1000)
result_50p = result_50p * 100
print("%f percent of the values are with the 0.6745 standard deviation" % result_50p)
print("")
# Integrate PDF from -2.698 to 2.698
result_99_3p, _ = quad(normalProbabilityDensity,
                     -2.698,
                     2.698,
                     limit = 1000)
result_99_3p = result_99_3p * 100
print("%f percent of the values are within 2.698 standard deviation" % result_99_3p)
print("")
#Outliers are the remaing 0.7 % of the data

# =============================================================================
# Boxplot on Real Data - Breast Cancer Wisconsin (Diagnostic) Dataset
# =============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Put dataset on my github repo 
df = pd.read_csv('https://raw.githubusercontent.com/mGalarnyk/Python_Tutorials/master/Kaggle/BreastCancerWisconsin/data/data.csv')

box = sns.boxplot(x='diagnosis', y='area_mean', data=df) # Seaborn version

malignant = df[df['diagnosis']=='M']['area_mean']
benign = df[df['diagnosis']=='B']['area_mean']

plt.figure(2)
plt.boxplot([malignant,benign], labels=['M', 'B'])


