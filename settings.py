#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anna SONG

settings.py allows one to

    - determine the dataset on which we perform the analysis:
        - landmarks
        - closed or open Hermite splines
        (- closed or open B-splines have not been used in the datasets, but the toolbox can be extended to B-splines too)
    
    - set the complex framework (C^N,Phi) (resp. the real framework (R^{2N},Psi))
    
    - convert the dataset to preshapes (centered and unit-norm configurations)
    
    - compute and show the Fréchet mean of the preshapes
    
Warning:
    
    Mathematical shapes can only be numerically handled as preshapes.
    
    Hence the `shape' variable designates in fact a preshape
    (centered and normalized configuration), that is one of the many representatives
    of the equivalence class defining the corresponding (mathematical) shape.


"""

import numpy as np
from numpy import loadtxt

import matplotlib.pyplot as plt
import colorsys
from splines_utils import giveHermitePhi, giveHermitePsi, fitHermiteSpline, drawHermiteSpline
from splines_utils import giveCubicBPhi, giveCubicBPsi, fitCubicBSpline, drawCubicBSpline


''' DATASET '''

data_type_names = ['landmarks','closed Hermite splines','open Hermite splines','closed cubic B-splines']

data_type = 0 # CHANGE HERE

print('NOW USING',data_type_names[data_type])

if data_type == 0 :
    choice = 0 # CHANGE HERE
    database = ['worms','hands','leaves_sym','mpeg7','mpeg7_sampled100','horses','900_horses_leaves_worms']
elif data_type == 1 :
    choice = 0 # CHANGE HERE
    database = ['50_open_worms_6ctrlpts','200_open_worms_6ctrlpts','6376_open_worms_6ctrlpts']
elif data_type == 2 :
    choice = 0 # CHANGE HERE
    database = ['200_closed_worms_8ctrlpts','100_closed_worms_10ctrlpts']
elif data_type == 3 :
    choice = 0 # CHANGE HERE
    database = ['leaves_Bsplines']


filename = 'DATA/' + database[choice] + '.csv'
'loading real configurations in R^{2N} (not necessarily preshapes at this stage!)'
shapes_r = loadtxt(filename,delimiter=',') 
K = len(shapes_r)

'test_k_set : the data to show when reconstructing with the dictionary and weights'
delta_k = K//4
test_k_set = [i*delta_k for i in range(4)]

if database[choice] in ['leaves_sym','leaves_Bsplines'] :
    test_k_set = [0,10,20,30] # for more interesting shapes to reconstruct
    

'N : dimension (a configuration z in C^N)'
'Phi (resp. Psi) : matrices determining the Hermitian (resp. real) scalar products'
if data_type == 0 :
    M = shapes_r.shape[1] // 2 # number of control points (here the same as the number of points)
    N = M
    # N equal to...
    # 20 for worms, 50 for hands, 100 for mpeg7_100 and 200 for leaves_sym, mpeg7, horses, 900_horses_leaves_worms
    Phi,Psi = np.eye(N),np.eye(2*N) 

elif data_type in [1,2] :
    periodic = data_type - 1
    M = shapes_r.shape[1] // 4 # number of control points
    N = 2*M # Hermite splines
    Phi,Psi = giveHermitePhi(M,periodic),giveHermitePsi(M,periodic) # Hermite splines

elif data_type == 3 :
    M = shapes_r.shape[1] // 2 # number of control points
    N = M
    Phi,Psi = giveCubicBPhi(M),giveCubicBPsi(M) # cubic B splines
    

''' COMPLEX FRAMEWORK '''

def real2complex(config_r) :
    '''Converts a real configuration in R^{2N} to a complex configuration in C^N.'''
    return config_r[:N] + 1j*config_r[N:]

def multi_real2complex(configs_r) :
    '''Same as real2complex(), but for horizontally stacked real configurations.'''
    return configs_r[:,:N] + 1j*configs_r[:,N:]

def complex2real(config_c) :
    '''Converts a complex configuration in C^N to a real configuration in R^{2N}.'''
    return np.concatenate((config_c.real,config_c.imag))

def multi_complex2real(configs_c) :
    '''Same as complex2real(), but for horizontally stacked complex configurations.'''
    return np.concatenate((configs_c.real,configs_c.imag),axis = 1)

def norm2(z) :
    '''Squared norm of z in C^N w.r.t the Hermitian product Phi.'''
    return z.conj().T @ Phi @ z

def norm(z) :
    '''Norm of z in C^N w.r.t the Hermitian product Phi.'''
    return np.sqrt(norm2(z)).real

def her(z,w) :
    '''Hermitian product of z and w in (C^N,Phi)'''
    return z.conj().T @ Phi @ w

def locmean(z) :
    '''Computes the mean (landmarks: arithmetic mean; splines: temporal mean)
    of the configuration z in C^N.'''
    if data_type in [0,2,3] : # landmarks or closed Hermite splines or closed cubic B splines
        m = np.mean(z[:M])
    elif data_type == 1 : # open Hermite splines
        m = np.sum(z[1:M-1])/(M-1) + (z[0]+z[M-1])/(2*M-2) + (z[M]-z[-1])/(12*M-12)
    return m

def preshape(z) :
    '''Centers and normalizes (i.e. preshapes it) the configuration z in C^N.'''
    m = locmean(z)
    z = z - m*(np.arange(N) < M)
    s = norm(z)
    return z/s

def multi_preshape(configs_c) :
    '''Same as preshape(), but for several horizontally stacked configurations.'''
    shapes = np.zeros((configs_c.shape[0],N),dtype = complex)
    for k in range(configs_c.shape[0]) :
        shapes[k] = preshape(configs_c[k])
    return shapes

def mean(shapes) : # preshapes
    '''Input: dataset of preshapes called `shapes'.
    Output: Fréchet mean (w.r.t. the distance d_F) of the dataset of shapes.
    It is mathematically a shape, numerically handled as a preshape.'''
    SQ = shapes.T @ shapes.conj() @ Phi
    D,V = np.linalg.eig(SQ)
    ds = np.real(D)
    inds = np.argsort(ds)[::-1]
    m = V[:,inds[0]]
    m = preshape(m) # VERY IMPORTANT CONDITION
    return m


''' CONVERTING TO PRESHAPES (COMPLEX AND REAL) '''

shapes = multi_real2complex(shapes_r) # complex configurations in C^N (not necessarily preshaped)
shapes = multi_preshape(shapes) # complex preshapes
original_shapes_r = shapes_r # saving the original shapes_r
shapes_r = multi_complex2real(shapes) # so that we get preshaped real configurations

'defining the normalized unit configuration'
# These three lines are suitable for landmarks as well as splines.
uu = np.zeros(N,dtype = complex)
uu[:M] = 1
uu = uu / norm(uu)


''' DISPLAY FUNCTIONS '''

'z0 is some preshape along which all the results are aligned before being shown'
if data_type == 0 :
    z0 = np.arange(-1,1,2/N) + 0j*np.zeros(N) # straight from left to right
    z0 = z0 / norm(z0)
elif data_type == 1 :
    z_ctrl = np.arange(-1,1,2/M) + 0j*np.zeros(M)
    z_tgt = 0j*np.zeros(M)
    z0 = np.concatenate( (z_ctrl,z_tgt) )
    z0 = z0 / norm(z0)
elif data_type == 2 :
    z_ctrl = np.exp(1j*2*np.pi*np.arange(M)/M+1j*np.pi)
    z_tgt = 1j*z_ctrl/M
    z0 = np.concatenate( (z_ctrl,z_tgt) ) 
    z0 = z0 / norm(z0)
elif data_type == 3 :
    z0 = np.exp(1j*2*np.pi*np.arange(M)/M+1j*np.pi)
    z0 = z0 / norm(z0)



def drawLks(shape, continuous = False, i = -1, size = 50, comparing = False):
    'shape designates a configuration in C^N'
    Colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b','#d62728', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    col = 'k'
    if i >= 0 :
        col = Colors[np.mod(i,len(Colors))]
    
    if continuous or comparing :
        plt.plot(shape.real, shape.imag, color= col)
    else :
        plt.scatter(shape.real, shape.imag,marker = 'o', color= col, s = size)
    if comparing :
        head_color = Colors[7] # gray
    else :
        head_color = 'r'
    plt.scatter(shape.real[0],shape.imag[0],marker = 'o',color = head_color, s = 8/5*size)
    plt.axis('equal')
    #plt.show()


def drawMany(shapes, force = False, L = 5, show = True) :
    '''shapes designates stacked configurations in C^N.
    Displays the 20 first shapes (if force = True then displays all the elements by groups of 20).    
    L is the maximum number of figures to be displayed per line.'''
    K = shapes.shape[0]
    if K > 20 :
        if not force :
            print('displaying only the first 20 shapes')
            drawMany(shapes[:20],show = show)
        else :
            drawMany(shapes[:20],show = show)
            print('and then...')
            drawMany(shapes[20:],force = True,show = show)
    else :
        if data_type == 0 :    
            x = shapes.real
            y = shapes.imag
                    
            ext_x = np.max(np.abs(x))
            ext_y = np.max(np.abs(y))
            ext = 1.1*max(ext_x,ext_y)
  
            col_delta = int(360 / (1.2*K))
            plt.figure()
            
            for k in range(K) :
                l = np.mod(k,L) ; h = int(k/L)
                off_x = l*2*ext
                off_y = -h*2*ext
                col = list(colorsys.hsv_to_rgb(np.mod(230 + col_delta*k,360)/360,0.7,0.7))
                if database[choice] == 'hands' or N >= 100 :
                    plt.plot(off_x + x[k], off_y + y[k],color = col)
                else :
                    plt.scatter(off_x + x[k], off_y + y[k],marker = 'o',color = col, s = 2)
                plt.scatter(off_x + x[k,0],off_y + y[k,0],marker = 'o',color = 'r', s = 20)
                    
                    
        elif data_type in [1,2] :
            periodic = data_type - 1
            ck = np.concatenate((shapes[:,:M,None].real,shapes[:,:M,None].imag),axis = 2)
            dk = np.concatenate((shapes[:,M:,None].real,shapes[:,M:,None].imag),axis = 2)
            
            ext_x = np.max(np.abs(ck[:,:,0]))
            ext_y = np.max(np.abs(ck[:,:,1]))
            ext = 1.1*max(ext_x,ext_y)

            col_delta = int(360 / (1.2*K))
            plt.figure()
            for k in range(K) :
                l = np.mod(k,L) ; h = int(k/L)
                spline = fitHermiteSpline(ck[k],dk[k],periodic)
                off_x = l*2*ext
                off_y = -h*2*ext
                col = list(colorsys.hsv_to_rgb(np.mod(230 + col_delta*k,360)/360,0.7,0.7))
                plt.scatter(off_x + ck[k,:,0], off_y + ck[k,:,1],marker = 'o',color = col, s = 2)
                plt.scatter(off_x + ck[k,0,0],off_y + ck[k,0,1],marker = 'o',color = 'r', s = 20)
                plt.scatter(off_x + spline[:,0],off_y + spline[:,1],marker = '.',color = col, s = .1)

        elif data_type == 3 :
            ck = np.concatenate((shapes[:,:,None].real,shapes[:,:,None].imag),axis = 2)
            
            ext_x = np.max(np.abs(ck[:,:,0]))
            ext_y = np.max(np.abs(ck[:,:,1]))
            ext = 1.1*max(ext_x,ext_y)

            col_delta = int(360 / (1.2*K))
            plt.figure()
            
            for k in range(K) :
                l = np.mod(k,L) ; h = int(k/L)
                spline = fitCubicBSpline(ck[k])
                off_x = l*2*ext
                off_y = -h*2*ext
                col = list(colorsys.hsv_to_rgb(np.mod(230 + col_delta*k,360)/360,0.7,0.7))
                plt.scatter(off_x + ck[k,:,0], off_y + ck[k,:,1],marker = 'o',color = col, s = 2)
                plt.scatter(off_x + ck[k,0,0],off_y + ck[k,0,1],marker = 'o',color = 'r', s = 20)
                plt.scatter(off_x + spline[:,0],off_y + spline[:,1],marker = '.',color = col, s = .1)


        plt.axis('equal')
        plt.axis('off')
        if show : plt.show()


if data_type == 0 :
    if N < 100 :
        draw = drawLks
    if database[choice] == 'hands' or N >= 100  :
        def draw(shape,i = -1,size = 50,comparing = False):
            return drawLks(shape, continuous = True,i = i,size = size,comparing = comparing)
        
elif data_type in [1,2] :
    periodic = data_type - 1
    def draw(shape,i = -1, comparing = False) :
        return drawHermiteSpline(shape,periodic,i = i,doweshow = True, comparing = comparing)

elif data_type == 3 :
    def draw(shape,i = -1, comparing = False) :
        return drawCubicBSpline(shape,i = i,doweshow = True, comparing = comparing)


''' SHOWING THE FRECHET MEAN OF THE DATASET '''

m = mean(shapes) # Fréchet mean of the shapes for the Full Procrustes distance d_F
draw(m) ; plt.title('Fréchet mean w.r.t. d_F') ; plt.show()
print('There are {} shapes.'.format(K))


if data_type == 0 :
    text = 'LANDMARKS '
elif data_type == 1 :
    text = 'OPEN Hermite SPLINES '
elif data_type == 2 :
    text = 'CLOSED Hermite SPLINES '
elif data_type == 3 :
    text = 'CLOSED Cubic B-SPLINES '
    
print('DATASET USED: ' + text + database[choice]+'\n')


if __name__ == '__main__' :
    drawMany(shapes)
