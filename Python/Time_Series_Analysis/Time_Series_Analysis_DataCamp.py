# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:14:10 2020

@author: Laura

Time Series in Python 

Goals
    * Learn about time series
    * Fit data to a time series model
    * Use the models to make forecasts of the future
    * Learn to use the relevant statistical packages in Python

Pandas Tools
    * df.index = pd.to_datetime(df.index), change the index to datetime
    * df.plot(), plot the data
    * df1.join(df2), join two dataframes
    * df['col'].pct_change and df['col'].diff(), compute percentage changes 
      and differences of a time series
    * df['ABC'].corr(df['XYZ']), correlation method
"""
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
# Google Trends Time Series, focus area 'diet'
# =============================================================================
diet = pd.read_csv('Google_Trends_Diet.csv', skiprows=1)

"""
'diet'' throughout the calendar year, hitting a low around the December 
holidays, followed by a spike in searches around the new year as people make 
New Year's resolutions to lose weight.
"""

# convert the indexes of dates of the 'diet' to datatime
diet.index = pd.to_datetime(diet.index) 

# Plot the entire time series diet and show gridlines
diet.plot(grid=True)
plt.show()

# Let look only on the 2012 data
diet2012 = diet['2012']

# Plot 2012 data
diet2012.plot(grid=True)
plt.show()


