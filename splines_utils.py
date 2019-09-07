#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anna SONG

This auxiliary script computes the Phi matrix, such that C^N endowed with Phi is isometric
to the subspace of splines of a certain type with n degrees of freedom.

Phi is in fact the Gram matrix of the basis functions phi_n

- closed Hermite splines
- open Hermite splines
- closed cubic B-splines 

We also define the corresponding display functions.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

        
''' SPLINE GENERATOR FUNCTIONS '''

def phi1(x) :
    'for Hermite spline'
    if (0 <= x)*(x <= 1) :
        return (1.0 + (2.0 * x)) * (x - 1)**2
    elif -1 <= x and x < 0 :
        return (1.0 - (2.0 * x)) * (x + 1)**2
    else :
        return 0.0

def phi2(x) :
    'for Hermite spline'
    if 0 <= x and x <= 1 :
        return x * (x - 1)**2
    elif -1 <= x and x < 0 :
        return x * (x + 1)**2
    else :
        return 0.0

def beta3(x) :
    'for cubic B-spline'
    if np.abs(x) < 1 :
        return 2/3 - x**2 + np.abs(x)**3/2
    if np.abs(x) < 2 :
        return (2-np.abs(x))**3/6
    else :
        return 0.0
    
    
''' FOR HERMITE SPLINES (valid for M big enough) '''

def perPhi1Phi1(I,M) :
    # Supposing M > 2
    A = np.mod(I,M)
    Phi11 = np.zeros(I.shape)
    Phi11[A == 0] = 26.0/(35*M)
    Phi11[A == 1] = 9.0/(70*M)
    Phi11[A == M-1] = 9.0/(70*M)
    return Phi11

def perPhi2Phi2(I,M) :
    # Supposing M > 2
    A = np.mod(I,M)
    Phi22 = np.zeros(I.shape)
    Phi22[A == 0] = 2.0/(105*M)
    Phi22[A == 1] = -1.0/(140*M)
    Phi22[A == M-1] = -1.0/(140*M)
    return Phi22

def perPhi1Phi2(I,M) :
    # Supposing M > 2
    A = np.mod(I,M)
    Phi12 = np.zeros(I.shape)
    Phi12[A == 1] = 13./(420*M)
    Phi12[A == M-1] = -13./(420*M)
    return Phi12


def openPhi1Phi1(I,M) :
    # Supposing M > 2
    Phi11 = M/(M-1)*perPhi1Phi1(I,M)
    Phi11[0,0] = .5*Phi11[0,0]
    Phi11[-1,-1] = .5*Phi11[-1,-1]
    Phi11[-1,0] = 0
    Phi11[0,-1] = 0
    return Phi11
    
def openPhi2Phi2(I,M) :
    # Supposing M > 2
    Phi22 = M/(M-1)*perPhi2Phi2(I,M)
    Phi22[0,0] = .5*Phi22[0,0]
    Phi22[-1,-1] = .5*Phi22[-1,-1]
    Phi22[-1,0] = 0
    Phi22[0,-1] = 0
    return Phi22

def openPhi1Phi2(I,M) :
    # Supposing M > 2
    Phi12 = M/(M-1)*perPhi1Phi2(I,M)
    Phi12[0,0] = 11./(210*(M-1))
    Phi12[-1,-1] = -11./(210*(M-1))
    Phi12[0,-1] = 0
    Phi12[-1,0] = 0
    return Phi12


def giveHermitePhi(M,per):
    'for Hermite splines only (not B-splines)'
    II = np.repeat(np.arange(M)[:,None],M,axis = 1)
    JJ = np.repeat(np.arange(M)[None,:],M,axis = 0)
    if per :
        Phi11 = perPhi1Phi1(II-JJ,M)
        Phi22 = perPhi2Phi2(II-JJ,M)
        Phi12 = perPhi1Phi2(II-JJ,M)
    else :
        Phi11 = openPhi1Phi1(II-JJ,M)
        Phi22 = openPhi2Phi2(II-JJ,M)
        Phi12 = openPhi1Phi2(II-JJ,M)
    Phi = np.concatenate((np.hstack((Phi11,Phi12)),np.hstack((Phi12.T,Phi22))))
    return Phi
    
def giveHermitePsi(M,per) :
    'for Hermite splines only (not B-splines)'
    Phi = giveHermitePhi(M,per)
    Psi = np.concatenate((np.hstack((Phi,0*Phi)),np.hstack((0*Phi,Phi))))
    return Psi


def fitHermiteSpline(ck,dk,per,Ns=100) :
    '''Given control points ck, dk corresponding to positions and tangents,
    fitHermiteSpline() draws the continuous Hermite spline curve (open or closed). Of course, this curve is
    sampled at a rate Ns (nb of sample points between two consecutive control points).
    per = 0 indicates open, per = 1 indicates periodic (closed)'''
    M = ck.shape[0]
    if per :
        ck = np.vstack((ck,ck[0]))
        dk = np.vstack((dk,dk[0]))
        K = M
    else:
        K = M-1
    Ntot = K*Ns
    spline = np.zeros((Ntot,2))
    for i in range(Ntot) :
        k = int(np.floor(i/Ns))
        r = np.mod(i,Ns)
        x = ck[k,0]*phi1(r/Ns) + ck[k+1,0]*phi1(r/Ns - 1) + dk[k,0]*phi2(r/Ns) + dk[k+1,0]*phi2(r/Ns-1)
        y = ck[k,1]*phi1(r/Ns) + ck[k+1,1]*phi1(r/Ns - 1) + dk[k,1]*phi2(r/Ns) + dk[k+1,1]*phi2(r/Ns-1)
        spline[i,0] = x
        spline[i,1] = y
    return spline


def drawHermiteSpline(shape,per,i = -1,doweshow = True, comparing = False) :
    ''' If shape is a complex vector containing the positions and tangents at the
    control points, drawHermiteSpline() draws the continuous curve of the Hermite spline.
    '''
    M = len(shape) // 2
    Colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b','#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    col = 'k'
    if i >= 0 :
        col = Colors[np.mod(i,len(Colors))]
    if comparing :
        col = Colors[7]
        head_color = Colors[7]
    else :
        head_color = 'r'
              
    ck = np.zeros((M,2))
    dk = np.zeros((M,2))
    ck[:,0] = shape[:M].real
    ck[:,1] = shape[:M].imag
    dk[:,0] = shape[M:].real
    dk[:,1] = shape[M:].imag
    spline = fitHermiteSpline(ck,dk,per,Ns=100)
    plt.scatter(ck[:,0],ck[:,1],marker = 'o',color = col, s = 50)
    plt.scatter(ck[0,0],ck[0,1],marker = 'o',color = head_color, s = 80)
    plt.scatter(spline[:,0],spline[:,1],marker = '.', color = col, s = 5)
    plt.axis('equal')
    #plt.show()
    

''' FOR CUBIC B-SPLINES '''

def L2sca(u,v,show = False) :
    'defines the scalar product of u and v on [0,1] '
    integ = quad(lambda t : u(t)*v(t),0,1)
    if show : print(integ)
    return integ[0]
# (297/1260+1/252)*2 = quad(lambda t : beta3(t)**2,-2,2)


def giveCubicBPhi(M):
    'for closed cubic B-splines only (not open ones)'
    Phi = np.zeros((M,M))
    def aux(k):
        return lambda t : beta3(M*(t-1) - k) + beta3(M*t - k) + beta3(M*(t+1) - k)
    for k in range(M):
        for l in range(M):
            u = aux(k)
            v = aux(l)
            Phi[k,l] = L2sca(u,v)
    return Phi

def giveCubicBPsi(M) :
    'for closed cubic B-splines only (not open ones)'
    Phi = giveCubicBPhi(M)
    Psi = np.concatenate((np.hstack((Phi,0*Phi)),np.hstack((0*Phi,Phi))))
    return Psi


def fitCubicBSpline(ck,Ns=20) :
    'For closed cubic B-splines only (not open ones)'
    '''Given the positions ck of control points, fitCubicBSpline() draws
    the continuous closed cubic B-spline curve, sampled at a rate Ns
    (nb of sample points between two consecutive control points).
    '''
    M = ck.shape[0]
    Ntot = M*Ns
    spline = np.zeros((Ntot,2))
    for i in range(Ntot) :
        k = int(np.floor(i/Ns))
        r = np.mod(i,Ns)
        x = ck[np.mod(k-1,M),0]*beta3(r/Ns + 1) + ck[np.mod(k,M),0]*beta3(r/Ns) + ck[np.mod(k+1,M),0]*beta3(r/Ns - 1) + ck[np.mod(k+2,M),0]*beta3(r/Ns - 2)
        y = ck[np.mod(k-1,M),1]*beta3(r/Ns + 1) + ck[np.mod(k,M),1]*beta3(r/Ns) + ck[np.mod(k+1,M),1]*beta3(r/Ns - 1) + ck[np.mod(k+2,M),1]*beta3(r/Ns - 2)
        spline[i,0] = x
        spline[i,1] = y
    return spline

def drawCubicBSpline(shape, i = -1,doweshow = True, comparing = False) :
    'For closed cubic B-splines only (not open ones)'
    ''' If shape is a complex vector containing the positions of the
    control points, drawCubicBSpline() draws the continuous curve of the closed cubic B-spline.
    '''
    Colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b','#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    col = 'k'
    if i >= 0 :
        col = Colors[np.mod(i,len(Colors))]
    if comparing :
        col = Colors[7]
        head_color = Colors[7]
    else :
        head_color = 'r'
              
    ck = np.concatenate((shape.real[None],shape.imag[None])).T
    spline = fitCubicBSpline(ck,Ns=20)
    plt.scatter(ck[:,0],ck[:,1],marker = 'o',color = col, s = 20)
    plt.scatter(ck[0,0],ck[0,1],marker = 'o',color = head_color, s = 80)
    plt.scatter(spline[:,0],spline[:,1],marker = '.', color = col, s = 5)
    #plt.axis('equal')

