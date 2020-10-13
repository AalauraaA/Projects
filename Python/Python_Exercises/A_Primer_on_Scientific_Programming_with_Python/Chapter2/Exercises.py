# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 21:35:34 2020

@author: Laura
------------------------------------------------------------------------------
A Primer on Scientific Programming with Python - Chapter 2 - Exercises
------------------------------------------------------------------------------
"""
# =============================================================================
# Exercise 2.1 - Make a Fahrenheit-Celsius Conversion Table
# =============================================================================
print('------------------') #table heading
F = 0
dF = 10 # increment of F in loop

while F <= 100:
    C = (F - 32)/(9.0/5)  #celsius 
    #F = (9.0/5)*C + 32 # 1st statement inside loop
    print('Fahrenheit: {}, Celsius: {}'.format(F,C))
    F = F + dF
print('------------------') # end of table line (after loop)

print('------------------') #table heading
C = -20
dC = 5 # increment of F in loop

while C <= 40:
    F = (9.0/5)*C + 32 #fahrenheit 
    print('Fahrenheit: {}, Celsius: {}'.format(F,C))
    C = C + dC
print('------------------') # end of table line (after loop)

# =============================================================================
# Exercise 2.2 - Generate an Approximate Fahrenheit-Celsius Conversion Table
# =============================================================================
print('------------------') #table heading
F = 0
dF = 10 # increment of F in loop

while F <= 100:
    C = (F - 32)/(9.0/5)  #celsius 
    C_ = (F - 30)/2
    #F = (9.0/5)*C + 32 # 1st statement inside loop
    print('Fahrenheit: {}, Celsius: {}, C_: {}'.format(F,C,C_))
    F = F + dF
print('------------------') # end of table line (after loop)
print("")

# =============================================================================
# Exercise 2.3 - Work With a List
# =============================================================================
primes = [2, 3, 5, 7, 11, 13]
p = 17
primes.append(p)

for i in primes:
    print(i)
print('-------------------')
# =============================================================================
# Exercise 2.4 - Generate Odd Numbers
# =============================================================================
n = 1

while n <= 20:
    if n % 2 != 0:
        print(n)
    n = n + 1

print('--------------------')

# =============================================================================
# Exercise 2.5 - Sum of First n Integers
# =============================================================================




#Opgave 2.7 - ball_table1.py

v = 5.0         # Initial velocity of the ball
g = 9.81        # Acceleration of gravity
t = 0.0         # Time

end = 2.0*(v / g)
i = 0
step = end/11

print "t\t\ty(t)"
while t < end:    
    y = (v * t) - (0.5 * g * t**2)
    print "%f\t%f" % (t, y)
    t += step    

"""
#Svar:
t		y(t)
0.000000	0.000000
0.092670	0.421226
0.185340	0.758208
0.278009	1.010943
0.370679	1.179434
0.463349	1.263679
0.556019	1.263679
0.648689	1.179434
0.741359	1.010943
0.834028	0.758208
0.926698	0.421226
"""

#Opgave 2.8 - ball_table2.py

t = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

def y(t):
    # formula 1.1 - find the vertical position of a ball at a specific (t) time
    v0 = 5 # initial velocity
    g = 9.81 # gravity 
    
    return (v0*t)- (0.5*g*(t**2))

# calculate the y for the different t values
y = [y(T) for T in t]

i = 0
print '---------------------'
print '|__t______|y(t)_____|'

# loop through our two lists and print the values
while i < len(y):
    print '|  %g   |%4.4f    |' % (t[i], y[i])
    i = i+1
print '---------------------'

"""
#Svar:
---------------------
|__t______|y(t)_____|
|  0.1   |0.4509    |
|  0.2   |0.8038    |
|  0.3   |1.0585    |
|  0.4   |1.2152    |
|  0.5   |1.2737    |
|  0.6   |1.2342    |
---------------------
"""
#Opgave 2.10 - sum_while.py
#Opgave 2.14 - index_nested_list.py
















