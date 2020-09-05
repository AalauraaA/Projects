# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:24:47 2018

@author: laura
@Url: https://www.practicepython.org/exercise/2014/02/15/03-list-less-than-ten.html
"""

# =============================================================================
# Exercise 3: Take a list, say for example this one: 
# a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89] and write a program that prints 
# out all the elements of the list that are less than 5.
# =============================================================================
a_list = [1,1,2,3,5,8,13,21,34,55,89]

new_a = [a for a in a_list if a < 5]
print("The new list is:")
print(new_a)

number = input("Write a number: ")
user_a = [a for a in a_list if a < int(number)]
print("The the new list is:")
print(user_a)