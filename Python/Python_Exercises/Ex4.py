# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 18:30:20 2018

@author: Laura
@Url: https://www.practicepython.org/exercise/2014/02/05/02-odd-or-even.html
"""
# =============================================================================
# Exercise 4: Create a program that asks the user for a number and then prints 
# out a list of all the divisors of that number. (If you don’t know what a 
# divisor is, it is a number that divides evenly into another number. 
# =============================================================================

number = input("Write a number: ")

number_list = [num for num in range(1, int(number)+1) if int(number) % num == 0]
print("This is the amount of divisor your number has %s:" %number, number_list)