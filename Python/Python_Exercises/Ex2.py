# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 17:46:52 2018

@author: Laura
@Url: https://www.practicepython.org/exercise/2014/02/05/02-odd-or-even.html
"""

# =============================================================================
# Exercise 2: Ask the user for a number. Depending on whether the number is 
# even or odd, print out an appropriate message to the user.
# =============================================================================

number = input("Write a number: ")
check = input("Then another one: ")

def function(number, check):
    x = int(number)   # Even or Odd
    print("")
    if x % 2 == 0:
        print("Your first number %d is even" % int(number))
    else:
        print("Your first number %d is odd" % int(number))
    
    y = int(number) % 4  # Multiple of 4
    print("")
    if y == 0:
        print("Your first number %d is a multiple of 4" % int(number))
    else:
        print("Your first number %d is not a multiple of 4" % int(number))
    
    num = int(number)
    check = int(check)
    if num/check == 0:
        print("Your choice of numbers %d and %d do not give a division rest" % num, check)
    else:
        div = num/check
        print("Your choice of numbers %d and %d give the division rest %d" % (num,check,div))
        return
