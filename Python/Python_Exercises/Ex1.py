# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:30:36 2018

@author: Laura
@Url: https://www.practicepython.org/exercise/2014/01/29/01-character-input.html
"""
# =============================================================================
# Exercise 1:
# Create a program that asks the user to enter their name and their age. Print 
# out a message addressed to them that tells them the year that they will turn 
# 100 years old.
# =============================================================================
name  = input("Skriv dit navn: ")
age = input("Skriv din alder: ")
year   = input("Hvilket aar har vi nu?: ")

def Hundrede(age, year):
    year_difference = 100 - int(age)
    year100 = year_difference + int(year)
    return year100

year100 = Hundrede(age, year)

print("")
print('%s, paa %s, bliver 100 aar i %d' % (name, age, year100))
  

