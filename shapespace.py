#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anna SONG

shapespace.py allows one to work in Kendall's shape space (in the 2D case),
extended to spline curves thanks to the Hermitian product Phi.

shapespace.py relies on the main script settings.py
but is not used by our KSD algorithm.

It defines:
    
    - the three distances d_F, d_P, and rho, also called full, partial, or geodesic distances.
    - some operations on the Riemannian manifold, such as exp(z,v), log(z,w), geo(z,w),
      and logarithmic or orthogonal projections on a tangent space
      
All functions are defined supposing z and w to be preshapes.

Warning:
    
    Mathematical shapes can only be numerically handled as preshapes.
    
    Hence the `shape' variable designates in fact a preshape
    (centered and normalized configuration), that is one of the many representatives
    of the equivalence class defining the corresponding (mathematical) shape.

"""

import numpy as np
from numpy import cos, sin, sqrt, arccos, exp

from settings import N
from settings import draw, drawMany
from settings import norm, her, mean


def d_F(z,w) : # preshapes
    '''Full distance between [z] and [w]
    min_{alpha,theta} ||alpha e^{i theta} z - w||
                    = || w - P_z w || = ||z - P_w z|| \in [0,1] '''
    aux = np.abs(her(z,w))**2
    if aux > 1 : # due to numerical approximations, but mathematically should be <= 1
        return 0.0
    return sqrt(1 - aux)

def d_P(z,w) : # preshapes
    '''Partial distance between [z] and [w]
    min_{theta}  ||e^{i theta} z - w||'''
    aux = 2*np.abs(her(z,w))
    if aux > 2 : # same possibility as for d_F
        return 0.0
    return sqrt(2 - aux)

def expo(z,v) : # z preshape, v in C^n referring to a tangent vector
    '''Computes the exponential of v at z, that corresponds to a preshape'''
    t = norm(v)
    if t < 1e-16 :
        return z
    return cos(t)*z + sin(t)/t*v

def rho(z,w) : # preshapes
    '''Geodesic distance between [z] and [w]. '''
    aux = np.abs(her(z,w))
    if aux > 1 : # happens sometimes due to numerical errors
        aux = 1
    return arccos(aux)

def theta(z,w) : # preshapes
    '''Computes the optimal angle theta(z,w) = arg(z* Phi w)
       solution to      min_{theta}  ||e^{i theta} z - w||'''
    return np.angle(her(z,w))

def log(z,w) : # preshapes
    '''Computes a preshape pertaining to the shape (equivalence class) log_[z] ([w])
    where log is relative the shape space Sigma.'''
    ta = theta(z,w)
    
    def logPre(z,w) :
        '''Computes v = log_z(w) where log is relative to the preshape sphere \S.
        (Requires that z* Phi w > 0, because otherwise
        v does not satisfy Re(z* Phi v) = 0 in order to be on the tangent space T_z \S.
        As a consequence, expo(logPre(z,w)) would not be a preshape in \S.)
        '''
        ro = rho(z,w)
        return ro/sin(ro)*(w - cos(ro)*z)
    
    return logPre(z,exp(-1j*ta)*w)

def align_rot(dataset, mm = None) :
    '''Optimally rotate the shapes along mm if it is given,
    otherwise their Fréchet mean with respect to d_F. '''
    K = len(dataset)
    rotated = np.zeros_like(dataset)
    if mm is None :
        m = mean(dataset)
    else :
        m = mm
    for k in range(K) :
        ta = theta(dataset[k],m)
        rotated[k] = exp(1j*ta)*dataset[k]
    return rotated

def showExpo(z,v,aux = 1, T = 4):
    ''' Displays elements on the exponential curve exp_z(t*v) for time t in [0,1]'''
    Time = np.arange(T+1)/(aux*T)
    path = np.zeros((T,N),dtype = complex)
    
    for i in range(T) :
        t = Time[i]
        path[i] = expo(z,t*v)
    
    drawMany(path)
    
def geo(z,w,T = 5) : # preshapes
    '''Returns elements regularly spaced along the geodesic curve joining z to w (preshapes).'''
    ro = rho(z,w)
    Time = np.arange(T+1)/T

    ta = theta(z,w)
    path = 1/sin(ro)*(sin((1-Time[:,None])*ro)*exp(1j*ta)*z + sin(Time[:,None]*ro)*w)
    return path

def showGeo(z,w) : # preshapes
    '''Shows the geodesic curve returned by geo(z,w).
    The results are similar to showExpo(z,log(z,w)).'''
    path = geo(z,w)    
    T = len(path) - 1 # same as in geo(z,w)

    for i in range(T+1) :
        draw(path[i],i = i)  
    
    drawMany(path)

def tangProj(m,z) : # preshapes
    '''Orthogonally project z onto tangent space at m (considering the ambient space)'''
    if np.abs(her(m,z)) < 1e-2 :
        return z
    ta = theta(z,m)
    v_proj = exp(1j*ta)*(z - her(m,z)*m)
    return v_proj

def multi_tangProj(m,dataset) :
    '''Same as tangProj() but for several shapes.'''
    K = len(dataset)
    v_proj = np.zeros((K,N),dtype = complex)
    for k in range(K) :
        z = dataset[k]
        v_proj[k] = tangProj(m,z)
    global X
    X = v_proj.T
    print('\n m + v_k on the tangent space are')
    drawMany(m+X.T)
    return v_proj

def multi_tangLog(m,dataset) :
    '''Same as log() but for several shapes.'''
    K= len(dataset)
    v_log = np.zeros((K,N),dtype = complex)
    for k in range(K) :
        z = dataset[k]
        v_log[k] = log(m,z)
    print('\n m + v_k on the tangent space are')
    drawMany(m+v_log)
    return v_log

from settings import shapes