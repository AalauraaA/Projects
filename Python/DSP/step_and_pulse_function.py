# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:34:01 2020

@author: Laura
------------------------------------------------------------------------------
DIGITAL SIGNAL PROCESSING
------------------------------------------------------------------------------
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# DSP Special Functions
# =============================================================================
" Constant Function - x(t) = 1 and x[n] = 1 "
const_func = 1

" Step Function "
def step_function(t):
    #or n
    step_func = np.zeros(len(t))
    for i in range(len(t)):
        if i < 0:
            step_func[i] = 0
        else:
            step_func[i] = 1
    return step_func

t = np.linspace(-10, 10, 11)

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(t, step_function(t))
plt.xlabel(r'$t$')
plt.ylabel(r'$x(t)$')
plt.title(r'Step function of $x(t)$')
plt.xlim([-0.05, 0.05])

plt.subplot(2,1,2)
plt.stem(t, step_function(t))
plt.xlabel('$n$')
plt.ylabel(r'$x[n]$')
plt.title(r'Step function of $x[n]$')
plt.xlim([-0.05, 0.05])
plt.show()


" Pulse Function "
def pulse_function(t, _type = 'discrete'):
    " The impulse function "
    if _type == 'discrete':    # n
        pulse_func = np.zeros(len(t))
        for i in range(len(t)):
            if i == 0:
                pulse_func[i] = 1
            else:
                pulse_func[i] = 0
    
    #if _type == 'continuous':
        
    return pulse_func
    





