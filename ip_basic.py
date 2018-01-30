# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:39:30 2018

@author: numerical tours
"""

from nt_toolbox.general import *
from nt_toolbox.signal import *
from pylab import *

name = 'nt_toolbox/data/lena.png'
n = 256
M = load_image(name, n)
# show the image
imageplot(M[::-1,:])
# compute the low pass  Gaussian kernel
sigma = 5
t = concatenate((range(0, int(n/2) + 1), range(int(-n/2),-1 )))
[Y,X] = np.meshgrid(t,t)
h = exp(-(X**2 + Y**2) / (2.0 * float(sigma)**2))
h = h/sum(h)
imageplot(fftshift(h))
# compute the periodic convolution using FFTs
Mh = real(ifft2(fft2(M) * fft2(h)))
# display
imageplot(M, 'Image', [1,2,1])
imageplot(Mh, 'Blurred', [1,2,2])
# Gradient
G = grad(M)
imageplot(G[:,:,0],'d/dx', [1,2,1])
imageplot(G[:,:,1],'d/dy', [1,2,2])
# compute and display fourier transform
Mf = fft2(M)
Lf = fftshift(log(abs(Mf) + 1e-1))
imageplot(M, 'Image', [1,2,1])
imageplot(Lf, 'Fourier transform', [1, 2, 2])






















