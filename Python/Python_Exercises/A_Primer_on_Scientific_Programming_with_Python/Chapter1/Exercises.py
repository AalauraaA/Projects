# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:30:58 2020

@author: Laura
------------------------------------------------------------------------------
A Primer on Scientific Programming with Python - Chapter 1 - Exercises
------------------------------------------------------------------------------
"""
from math import pi, sin, cos, sqrt, exp
# =============================================================================
# Exercise 1.1 - Compute 1+1
# =============================================================================
print('1 + 1 is equal to: {}'.format(1 + 1))
print("")

# =============================================================================
# Exercise 1.2 - Write a Hello World Program
# =============================================================================
print('Hello, World!')
print("")

# =============================================================================
# Exercise 1.3 - Derive and Compute a Formula
# =============================================================================
sec_ = 10E09
min_ = sec_/60
hour_ = min_/60
day_ = hour_/24
week_ = day_/7
year_ = week_/52

sec_to_year = ((((sec_/60)/60)/24)/7)/52   #on one line

if year_ < 100 or sec_to_year < 100:
    print(r'$10^9$ seconds is {} years'.format(year_)) 
    print("")
else:
    print('A baby can not be {} years old'.format(year_))
    print("")

# =============================================================================
# Exercise 1.4 - Convert From Metes to British Lenght Units
# =============================================================================
meter_ = 640

" Initial Measures "
one_inch = 0.0254
one_foot = 12 * one_inch
one_yard = 3 * one_foot
one_mile = 1760 * one_yard

" Converted Measures "    
inch_ = meter_/one_inch
foot_ = inch_/one_foot
yard_ = foot_/one_yard
mile_ = yard_/one_mile

print('{} meters correspond to {} inches, {} feet, {} yards, or {} miles'.format(meter_, inch_, foot_, yard_, mile_))
print("")
#a length of 640 meters corresponds to 25196.85 inches, 2099.74 feet, 699.91 yards, or 0.3977 miles.

# =============================================================================
# Exercise 1.5 - Compute the Mass of Various Substances
# =============================================================================

# =============================================================================
# Exercise 1.6 - Compute the Growth of Money in a Bank
# =============================================================================
A = 5000    #inital amount of money in euros
p = 0.05    #percent rate
n = 3       #year of growth

print('{} euros has grown into to {}'.format(A, A*(1 + (p/100))**n))
print("")

# =============================================================================
# Exercise 1.7 - Find Error(s) in a Program    
# =============================================================================
#x=1; print ’sin(%g)=%g’ % (x, sin(x))
x = 1
print(r'$sin({})$ = {}'.format(x, sin(x)))
print("")

# =============================================================================
# Exercise 1.8 - Type in Program Text
# =============================================================================
h = 5.0 # height
b = 2.0 # base
r = 1.5 # radius

area_parallelogram = h*b
area_square = b**2
area_circle = pi*r**2
volume_cone = 1.0/3*pi*r**2*h

print('The area of the parallelogram is %.3f' % area_parallelogram)
print("")
print('The area of the square is %g' % area_square)
print("")
print('The area of the circle is %.3f' % area_circle)
print("")
print('The volume of the cone is %.3f' % volume_cone)

# =============================================================================
# Exercise 1.9 - Type in Programs and Debug Them
# =============================================================================
#a) 
x = pi/4
val = sin(x)**2 + cos(x)**2
print(val)

#b)
v0 = 3
t = 1
a = 2
s = v0 * t + 0.5 * a * t**2
print(s)

#c)
a = 3.3 
b = 5.3

a2 = a**2
b2 = b**2

eq1_sum = a2 + 2*a*b + b2
eq2_sum = a2 - 2*a*b + b2

eq1_pow = (a + b)**2
eq2_pow = (a - b)**2

print('First equation: {} = {}'.format(eq1_sum, eq1_pow))
print('Second equation: {} = {}'.format(eq2_pow, eq2_pow))
print("")

# =============================================================================
# Exercise 1.10 - Evaluate a Gaussian Function
# =============================================================================
m = 0
s = 2
x = 1

gauss = 1/(sqrt(2 * pi) * s) * exp(-1/2 * ((x - m)/s)**2)

print('The Gaussian function is: {}'.format(gauss))
print("")

# =============================================================================
# Exercise 1.11 - Compute the Air Resistance on a Football
# =============================================================================
rho = 1.2             #density for air in kg m^-3
a = 11                #radius in cm
V_soft = 10           #velocity in km/h
V_hard = 120          #velocity in km/h
m = 0.43              #mass in kg
g = 9.81              #m s^-2
A = pi * a**2     
CD = 0.2              #drag coefficient


d_force_soft = 1/2 * CD * rho * A * V_soft**2
d_force_hard = 1/2 * CD * rho * A * V_hard**2
g_force = m * g       #gravity force in N = kg m/s^2

print("For a hard kick the drag force is {}".format(d_force_hard))
print("For a soft kick the drag force is {}".format(d_force_soft))
print("")






