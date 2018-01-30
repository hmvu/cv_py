# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:53:03 2018

@author: numerical tours

Image Approximation with Fourier and Wavelets
"""
from __future__ import division

import numpy as np
import scipy as scp
import pylab as pyl
import matplotlib.pyplot as plt

from nt_toolbox.general import *
from nt_toolbox.signal import *

import warnings
warnings.filterwarnings('ignore')
# load image
n0 = 512
f = rescale(load_image('nt_toolbox/data/lena.bmp', n0))
plt.figure(figsize=(5,5))
imageplot(f,'Image f')
# blurred by low pass kernel convolution
k = 9 # kernel size
h = np.ones([k,k])
h = h/np.sum(h) # normalize
from scipy import signal
fh = signal.convolve2d(f, h, boundary='symm')
plt.figure(figsize=(5,5))
imageplot(fh, 'Blurred image')
# Fourier transform
F = pyl.fft2(f)/n0
# check the energy
from pylab import linalg
print('Energy of Image: %f' %linalg.norm(f))
print('Energy of Fourier: %f' %linalg.norm(F))
# compute the logarithm of the Fourier magnitude
L = pyl.fftshift(np.log(abs(F) + 1e-1))
# display
plt.figure(figsize=(5,5))
imageplot(L, 'Log(Fourier transform)')
# Linear Fourier Approximation
# number of M of kept coefficient
M = n0**2//64
q = int(np.sqrt(M))
F = pyl.fftshift(pyl.fft2(f))
Sel = np.zeros([n0,n0])

Sel[n0//2 - q//2:n0//2 + q//2, n0//2 - q//2:n0//2 + q//2] = 1
F_zeros = np.multiply(F,Sel)

fM = np.real(pyl.ifft2(pyl.fftshift(F_zeros)))
plt.figure(figsize = (5,5))
imageplot(clamp(fM), "Linear, Fourier, SNR = %.1f dB" %snr(f, fM))
# compare
plt.figure(figsize=(7,6))
plt.subplot(2,1,1)
plt.plot(f[:, n0//2])
plt.xlim(0, n0)
plt.title('f')

plt.subplot(2,1,2)
plt.plot(fM[:, n0//2])
plt.xlim(0, n0)
plt.title('f_M')

# Non-linear Fourier Approximation: keeping M largest coeffs
T = .2 # threshold
F = pyl.fft2(f)/n0
FT = np.multiply(F, (abs(F) > T)) # filter
L = pyl.fftshift(np.log(abs(FT) + 1e-1))
plt.figure(figsize=(5,5))
imageplot(L, 'thresholed Log(Fourier transform)')
# inverse
fM = np.real(pyl.ifft2(FT)*n0)
plt.figure(figsize=(5,5))
imageplot(clamp(fM),'Non-Linear Fourier, SNR = %.1f dB'%snr(f,fM))
# given number of non-zero coefficient
m = np.sum(FT != 0)
print('M/N = 1/%d'%(n0**2/m))





























































