# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:43:42 2020

@author: Laura

Data Visualization in Python
"""
# =============================================================================
# Import of Visualization Libraries
# =============================================================================
from matplotlib import pyplot as plt

# =============================================================================
# Line Plot
# =============================================================================
"""
Line graphs are helpful for visualizing how a variable changes over time. Some 
possible data that would be displayed with a line graph are:
    * average prices of gasoline over the past decade
    * weight of an individual over the past couple of months
    * average temperature along a line of longitude over different latitudes
We can also have multiple line plots displayed on the same set of axes. This 
can be very useful if we want to compare two datasets with the same scale and 
axis categories. 

Sometimes, we want to display two lines side-by-side, rather than in the same 
set of x- and y-axes. When we have multiple axes in the same picture, we call 
each set of axes a subplot. The picture or object that contains all of the 
subplots is called a figure. We can have many different subplots in the same 
figure, and we can lay them out in many different ways. We can think of our 
layouts as having rows and columns of subplots. We can create subplots using 
'.subplot()'. The command 'plt.subplot()' needs three arguments to be passed 
into it:
    * number of rows of subplots
    * number of columns of subplots
    * index of the subplot we want to create
We can customize the spacing between our subplots to make sure that the figure 
we create is visible and easy to understand. To do this, we use the 
'plt.subplots_adjust()' command. '.subplots_adjust()' has some keyword 
arguments that can move your plots within the figure:
    * left (the left-side margin). You can increase this number to make room 
    for a y-axis label
    * right (the right-side margin). You can increase this number to make room 
    for a figure or decrease to make room for a legend
    * bottom (the bottom margin). You can increase this number to make room 
    for a tick mark labels or an x-axis label
    * top (the top margin)
    * wspace. The horizontal space between adjancent subplots   
    * hspace. The vertical space between adjancent subplots     
              
The most used commands are:
    plt.plot(x_values, y_values, color='', linestyle='', marker='')
    plt.axis()
    plt.xlabel()
    plt.ylabel()
    plt.title()
    plt.legend(loc='')
    plt.show()
    plt.subplot()
    plt.subplots_adjust()
    plt.close('all')
    plt.figure(figsize=(width, height))
    plt.savefig('')
    plt.fill_between(alpha=)
    ax = plt.subplot()
    ax.set_xticks()
    ax.set_yticks
    ax.set_xticklabels(rotation=)
    ax.set_yticklabels(rotation=)
"""

# Simple line plot
days = [0, 1, 2, 3, 4, 5, 6]                 # Days of the week
money_spent = [10, 12, 12, 10, 14, 22, 24]   # Money spent
money_spent_2 = [11, 14, 15, 15, 22, 21, 12] # Your friend's money spent

plt.figure(1)
plt.plot(days, money_spent, color='green', linestyle='--')
plt.plot(days, money_spent_2, color='#AAAAAA',  marker='o')
plt.xlabel('Days')
plt.ylabel('Money')
plt.title('Money Spent on Lunch')
plt.legend(['My Money', 'Friend Money'])
plt.show()

# Simple subplot
plt.figure(2)
# Data sets
x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

plt.subplot(1, 2, 1)
plt.plot(x, y, color='green')
plt.title('First Subplot')

plt.subplot(1, 2, 2)
plt.plot(x, y, color='steelblue')
plt.title('Second Subplot')

plt.subplots_adjust(wspace=0.35)
plt.show()

# Expanded subplot
plt.figure(3)
x = range(7)
straight_line = [0, 1, 2, 3, 4, 5, 6]
parabola = [0, 1, 4, 9, 16, 25, 36]
cubic = [0, 1, 8, 27, 64, 125, 216]

plt.subplot(2, 1, 1)
plt.plot(x, straight_line)

plt.subplot(2, 2, 3)
plt.plot(x, parabola)

plt.subplot(2, 2, 4)
plt.plot(x, cubic)
plt.subplots_adjust(wspace=0.35, bottom=0.2)
plt.show()

# Text line plot
plt.figure(4)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep","Oct", "Nov", "Dec"]
months = range(12)
conversion = [0.05, 0.08, 0.18, 0.28, 0.4, 0.66, 0.74, 0.78, 0.8, 0.81, 0.85, 0.85]

plt.xlabel("Months")
plt.ylabel("Conversion")
plt.plot(months, conversion)
plt.show()

# 
ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names)
ax.set_yticks([0.10, 0.25, 0.5, 0.75])
ax.set_yticklabels(["10%", "25%", "50%", "75%"])
plt.show()

# Shade line plot
plt.figure(5)
revenue = [16000, 14000, 17500, 19500, 21500, 21500, 22000, 23000, 20000, 19500, 18000, 16500]

plt.plot(months, revenue)
ax = plt.subplot()
ax.set_xticks(months)
ax.set_xticklabels(month_names)
y_lower = [i - 0.1 * i for i in revenue]
y_upper = [i + 0.1 * i for i in revenue]
plt.fill_between(months, y_lower, y_upper, alpha=0.2) #this is the shaded error
plt.show()

" Sublime Limes' Line Graphs - Project "
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

visits_per_month = [9695, 7909, 10831, 12942, 12495, 16794, 14161, 12762, 12777, 12439, 10309, 8724]

# numbers of limes of different species sold each month
key_limes_per_month = [92.0, 109.0, 124.0, 70.0, 101.0, 79.0, 106.0, 101.0, 103.0, 90.0, 102.0, 106.0]

persian_limes_per_month = [67.0, 51.0, 57.0, 54.0, 83.0, 90.0, 52.0, 63.0, 51.0, 44.0, 64.0, 78.0]

blood_limes_per_month = [75.0, 75.0, 76.0, 71.0, 74.0, 77.0, 69.0, 80.0, 63.0, 69.0, 73.0, 82.0]

x_values = range(len(months))

plt.figure(figsize=(12,8))

ax1 = plt.subplot(1,2,1)
plt.plot(x_values, visits_per_month, marker='o')
plt.xlabel('Months')
plt.ylabel('Page Visits')
ax1.set_xticks(x_values)
ax1.set_xticklabels(months)
plt.title('Total Page Visits')

ax2 = plt.subplot(1,2,2)
plt.plot(x_values, key_limes_per_month, color ='green')
plt.plot(x_values, persian_limes_per_month, color ='orange')
plt.plot(x_values, blood_limes_per_month, color ='red')
ax2.set_xticks(x_values)
ax2.set_xticklabels(months)
plt.legend(['key', 'persian', 'blood'])
plt.ylabel('Number of Limes')
plt.title('Total Sales of Limes')
plt.savefig('Page_visits_and_lime_sale.png')
# plt.show()

# =============================================================================
# Bar Chart
# =============================================================================
"""
The 'plt.bar' function allows you to create simple bar charts to compare 
multiple categories of data. You call plt.bar with two arguments:
    * the x-values — a list of x-positions for each bar
    * the y-values — a list of heights for each bar

We can use a bar chart to compare two sets of data with the same types of axis
values. To do this, we plot two sets of bars next to each other, so that the 
values of each category can be compared.     

The commands are:
    - plt.bar(bottom=, yerr=, capsize=)
"""
drinks = ["cappuccino", "latte", "chai", "americano", "mocha", "espresso"]
sales1 =  [91, 76, 56, 66, 52, 27]
sales2 = [65, 82, 36, 68, 38, 40]

n = 1   # This is our first dataset (out of 2)
t = 2   # Number of datasets
d = 6   # Number of sets of bars
w = 0.8 # Width of each bar
store1_x = [t*element + w*n for element
             in range(d)]

plt.bar(store1_x, sales1)

n = 2   # This is our second dataset (out of 2)
t = 2   # Number of datasets
d = 6   # Number of sets of bars
w = 0.8 # Width of each bar
store2_x = [t*element + w*n for element
             in range(d)]

plt.bar(store2_x, sales2)
plt.show()

plt.bar(range(len(drinks)),
  sales1)
plt.bar(range(len(drinks)),
  sales2, bottom=sales1)

plt.legend(['Location 1', 'Location 2'])
plt.show()
ounces_of_milk = [6, 9, 4, 0, 9, 0]
error = [0.6, 0.9, 0.4, 0, 0.9, 0]
plt.bar(range(len(drinks)), ounces_of_milk, yerr=error, capsize=5)
plt.show()

# =============================================================================
# Pie Chart
# =============================================================================
"""
If we want to display elements of a data set as proportions of a whole, we can 
use a pie chart. 

The commando are:
    - plt.pie(labels=, autopct=)
    - plt.axis('equal')
"""
budget_data = [500, 1000, 750, 300, 100]
budget_categories = ['marketing', 'payroll', 'engineering', 'design', 'misc']

plt.pie(budget_data)
plt.axis('equal')
plt.legend(budget_categories, autopct='%0.1f%%')
plt.show()

# =============================================================================
# Histogram
# =============================================================================
"""
Sometimes we want to get a feel for a large dataset with many samples beyond 
knowing just the basic metrics of mean, median, or standard deviation. To get 
more of an intuitive sense for a dataset, we can use a histogram to display 
all the values. A histogram tells us how many values in a dataset fall between 
different sets of numbers. All bins in a histogram are always the same size. 
The width of each bin is the distance between the minimum and maximum values 
of each bin. Each bin is represented by a different rectangle whose height is '
the number of elements from the dataset that fall within that bin.

Another problem we face is that our histograms might have different numbers of 
samples, making one much bigger than the other. We can normalize our histograms 
using normed=True. This command divides the height of each column by a constant 
such that the total shaded area of the histogram sums to 1.

The commandos are:
    - plt.hist(bins=, range=, alpha=, histtype='step', size=, normed=)
"""
import numpy as np
a = np.random.normal(loc=64, scale=2, size=10000)
b = np.random.normal(loc=70, scale=2, size=100000)

plt.hist(a, range=(55, 75), bins=20, alpha=0.5, normed=True)
plt.hist(b, range=(55, 75), bins=20, alpha=0.5, normed=True)
plt.show()


