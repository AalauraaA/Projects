# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:14:16 2020

@author: Laura
------------------------------------------------------------------------------
DIGITAL SIGNAL PROCESSING - SIGNALS and SOUNDS
------------------------------------------------------------------------------
A signal represents a quantity that varies in time. A type of signal is 
sound. A sound signal represents variations in air pressure over time.

A microphone is a device that measures these variations and generates an
electrical signal that represents sound. A speaker is a device that takes an
electrical signal and produces sound. Together they are called transducers.

Periodic signals are signal that repeats themselves after some period of time.
A signal is periodic if there are repeatly cycles of same duration. The duration
of each cycle is called a period.

The frequency of a signal is the number of cycles per second (the inverse of period)
and is measured by the unit hertz (Hz).

The shape of a periodic signal is called the waveform.
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Real Value Signals
# =============================================================================
" Continuous time "
t = np.linspace(-0.05, 0.05, 1000)       # time
f = 50                                   # frequency in Hz
A = 325                                  # amplitude  
func_t = A * np.sin(2 * np.pi * f * t)   # function of time (sinusiod)

plt.figure(1)
plt.plot(t, func_t)
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title(r'Plot of CT signal $x(t)= 325 * \sin(2 * \pi * 50 * t)$')
plt.xlim([-0.05, 0.05])
plt.show()

" Discrete time "
n = np.arange(50)                      # sample
fs = 0.07/50                           # sampling rate
x = np.sin(2 * np.pi * 50 * n * fs)    # Discrete-Time function of n

plt.figure(2)
plt.stem(n, x)
plt.xlabel('n')
plt.ylabel('x[n]')
plt.title(r'Plot of DT signal $x[n] = 325 * \sin(2 * \pi * 50 * n * \Delta t)$')
plt.show()

# =============================================================================
# Complex Value Signals
# =============================================================================
t = np.linspace(-0.05, 0.05, 1000)
func_com_t = np.exp(2j*np.pi*50*t)

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(t,func_com_t.real)
plt.xlabel('t')
plt.ylabel('Re x(t)')
plt.title(r'Real part of $x(t)=e^{j * 100 * \pi * t}$')
plt.xlim([-0.05, 0.05])

plt.subplot(2,1,2)
plt.plot(t, func_com_t.imag);
plt.xlabel('t')
plt.ylabel('Im x(t)')
plt.title(r'Imaginary part of $x(t) = e^{j * 100 * \pi * t}$')
plt.xlim([-0.05, 0.05])
plt.show()

plt.figure(4)
plt.subplot(2,1,1)
plt.plot(t, np.abs(func_com_t))
plt.xlabel(r'$t$')
plt.ylabel(r'$|x(t)|$')
plt.title(r'Absolute value of $x(t) = e^{j * 100 * \pi * t}$')
plt.xlim([-0.05, 0.05])

plt.subplot(2,1,2)
plt.plot(t, np.angle(func_com_t)*360/(2*np.pi))
plt.xlabel('$t$')
plt.ylabel(r'$\angle x(t)$')
plt.title(r'Phase of $x(t) = e^{j * 100 * \pi * t}$')
plt.xlim([-0.05, 0.05])
plt.show()