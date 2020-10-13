# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 15:30:05 2020

@author: Laura
------------------------------------------------------------------------------
DIGITAL SIGNAL PROCESSING - SPECTRAL DECOMPOSITION
------------------------------------------------------------------------------
Spectral decomposition is the idea that any signal can be expressed as a sum 
of sinusiod with different frequencies.

A discrete Fourier transform (DFT) takes the signal and produces its spectrum
The spectrum is the set of sinusoids that add up to produce the signal.
The Fast Fourier transfomr (FFT) is a more efficient way to use DFT.

When looking at a spectrum the x-axis shows the ranges of frequencies while
the y-axis shows the strenght or amplitude of each frequency

The lowest frequency component is called the fundamental frequency, the 
frequency with the largest amplitude is the dominant frequency. The other
are called harmonics
"""
import numpy as np
import matplotlib.pyplot as plt


" Periodic Sinus Signal "
t = np.linspace(0,5,1000)
f = np.sin(2 * np.pi * t)

t_fre = np.fft.fftfreq(t.shape[-1])
f_dft = np.fft.fft(t)

plt.figure(1)
plt.plot(t,f)

plt.figure(2)
plt.plot(t_fre, f_dft)
