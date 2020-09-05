# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 10:38:11 2020

@author: Laura

Recommendation System in Python
"""
import pandas as pd
import numpy as np

# =============================================================================
# Import of data files
# =============================================================================
frame = pd.read_csv('rating_final.csv')
cuisine = pd.read_csv('chefmozcuisine.csv')

" Looking at the data "
print(frame.head())   # Looking at the first 5 entries of the dataset frame
print("")
print(cuisine.head()) # Looking at the first 5 entries of the dataset cuisine
print("")

# =============================================================================
# Recommendation based on count
# =============================================================================
rating_count = pd.DataFrame(frame.groupby('placeID')['rating'].count())
# Sort the count from high to low
print(rating_count.sort_values('rating', ascending=False).head())

# Lets find the cuisine of the most rated places. By taking the ID from the 
# frame dataset, I can find the cuisine corresponding to the ID in the 
# cuisine dataset.
most_rated_places = pd.DataFrame([135085, 132825, 135032, 135052, 132834], index=np.arange(5), columns=['placeID'])
summary = pd.merge(most_rated_places, cuisine, on='placeID')
print(summary)
print("")

# Lets look at the statistical summary of Rcuisine
print(cuisine['Rcuisine'].describe())
print("")

# =============================================================================
# Recommendation based on correlation
# =============================================================================
# By introducing the geodata dataset for the frame and cuisine datasets I can 
# make some recommendation based on correlations.
geodata = pd.read_csv('geoplaces2.csv', encoding = 'mbcs')
print(geodata.head())
print("")

# I'm are only interested in the places of the ID from the frame data set
places =  geodata[['placeID', 'name']]
print(places.head())
print("")

# By grouping and ranking the data I can get a better view of the rating of 
# the restaurants
rating = pd.DataFrame(frame.groupby('placeID')['rating'].mean()) 
rating['rating_count'] = pd.DataFrame(frame.groupby('placeID')['rating'].count()) # add a rating_count columns to the mean rating
print(rating.describe())
print("")

print(rating.sort_values('rating_count', ascending=False).head())
print("")

# From the sorting I can see that 135085 have the highst rating count. Lets 
# look at what placed is connected to this ID.
print(places[places['placeID']==135085]) # place
print("")
cuisine[cuisine['placeID']==135085]      # cuisine
print("")

# =============================================================================
# Preparing for data analysis
# =============================================================================
places_crosstab = pd.pivot_table(data=frame, values='rating', index='userID', columns='placeID')
print(places_crosstab.head())
print("")

# I will now find all the rating connected to the restaurant 'Tortas Locas Hipocampo'
Tortas_ratings = places_crosstab[135085]
print(Tortas_ratings[Tortas_ratings>=0])
print("")

# With the data and the found rating I can evaluate the data on similarity to 
# the tortas restaurant based on correlation
similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings) # find ratings

corr_Tortas = pd.DataFrame(similar_to_Tortas, columns=['PearsonR'])
corr_Tortas.dropna(inplace=True)
print(corr_Tortas.head())
print("")

Tortas_corr_summary = corr_Tortas.join(rating['rating_count'])
print(Tortas_corr_summary[Tortas_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(10))
print("")

# I now know the placed correlated with 'Tortas Locas Hipocampo' and I will
# then look at what the cuisines are of those places
places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns=['placeID'])
summary = pd.merge(places_corr_Tortas, cuisine,on='placeID')
print(summary)
print("")

# The first entry is actual the restaurant I want to find as it is similar cuisine
# but entry 4 (the last entry) is also a cuisine I was looking for. Lets look 
# at the entry 4 name
print(places[places['placeID']==135046])
print("")











