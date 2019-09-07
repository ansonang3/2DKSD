#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:41:21 2018

@author: Anna SONG

KSD.py implements the `2D Kendall Shape Dictionary' that we propose in
[Song, Uhlmann, Fageot & Unser (2019)].
Given a dataset of preshapes z_1,...,z_K in (C^N,Phi) endowed with the Hermitian
product Phi, the original problem to be minimized by

        the dictionary D = [d_1,...,d_J] \in \C^{N \times J}
        the weights A = [a_1,...,a_K] \in \C^{J \times K}
        
is written as

        min_{D,A} sum_k d_F([z_k],[D a_k])^2            (*)

    - d_F is the Full Procrustes distance
    - where the atoms d_j in C^N are constrained to be preshapes
    - the weights a_k in C^J are subject to |a_k|_0 <= N_0
    - and such that D a_k are preshapes.
    
As shown in our article, this can be reformulated as as simple L2 problem in the complex setting

        min_{D,A} sum_k || z_k - D a_k ||_Phi^2             (**)

    - where the atoms d_j in C^N are constrained to be preshapes
    - and the weights a_k in C^J are subject to |a_k|_0 <= N_0.
    
The solutions to the two minimization problems are equivalent up to a rescaling of the weights a_k.


KSD.py implements the minimization in the form of (**).
The 2D Kendall Shape Dictionary classically alternates between:
    
    - a sparse coding step : the weights A are updated using a Cholesky-based 
    Order Recursive Matching Pursuit (ORMP), as a direct adaptation to the
    complex setting of Mairal's implementation for the real setting in the SPAMS toolbox.
    - a dictionary update : following the Method of Optimal Directions (MOD),
    we update D as
    
            D <- [z_1,...,z_K] @ A^H @ (A @ A^H)^{-1}
            D <- Pi_S(D) (center and normalize all the atoms d_j)
            


Rmks:
    - there is no need for a gradient descent with respect to D, although the expression
    of the gradient can be explicited:
        
        Grad_D [ sum_k || z_k - D a_k ||_Phi^2 ]
        
        = 2 Phi (D @ sum_k a_k @ a_k^H - sum_k z_k @ a_k^H )
    
    - after the sparse coding step in (**), D a_k are not preshapes, but just centered.
    One can decide to rescale them so as to obtain preshapes.
    
    - the sparse coding step ensures that the reconstructions D a_k in (**) are optimally
    aligned to the data z_k (resp., optimally rotated in (*) if considering the corresponding
    preshapes).
    
    - in the dictionary update step:
        - if A @ A^H is not invertible, we use the
            SVD decomposition of A (see the article).
        - In practice, the preshaping step D <- Pi_S(D) is just the normalization of
            all the columns d_j (because the previous operation already centers them,
            since z_1,...,z_K are already centered).

"""


import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import os
import scipy
import random
import time

from settings import M, N, K, data_type, test_k_set, database, choice, shapes # dataset contained in shapes
from settings import draw, drawMany
from settings import Psi, Phi, norm, her
from settings import multi_complex2real

from shapespace import theta, align_rot



''' INITIALIZATION '''


def initializeD_c(J,dataset) :
    '''Initialize the dictionary: randomly copy the existing data.'''
    D_c = np.zeros((N,J),dtype = complex)
    
    indices = np.arange(K) ; random.shuffle(indices)
    for j in range(J) :
        k = indices[j]
        D_c[:,j] = dataset[k]

    return D_c
    

''' SPARSE CODING '''


def ORMP_cholesky(D_c,dataset,N0) : # columns of D_c must be preshapes
    '''Order Recursive Matching Pursuit with Cholesky-based optimisation,
    as in the SPAMS toolbox of Mairal et al. (2009).
    This is a direct adaptation of their code in C++ to the complex setting
    with a Hermitian product Phi.
    The columns of D_c must all be preshapes.
    '''    
#    start = time.time()

    J = D_c.shape[1]
    K = len(dataset)
    A_c = np.zeros((J,K),dtype = complex)    
    G = D_c.T.conj() @ Phi @ D_c
        
    for k in range(K) :
        z = dataset[k]
        vM = np.zeros((N0),dtype = complex)
        rM = np.zeros((N0),dtype=int)

        Un = np.zeros((N0,N0), dtype = complex)
        Undn = np.zeros((J,N0),dtype = complex)
        Gs = np.zeros((N0,J),dtype = complex)
        norm2 = np.ones(J)
        
        rM[:] = -1
        Rdn = D_c.T.conj() @ Phi @ z
        scores = Rdn.copy()
        
        for l in range(N0) :
            currind = np.argmax(np.abs(scores))
            if np.abs(scores[currind]) < 1e-8 :
                #print('finished with a sparsity l =',l,'for data',k)
                break
            
            RU = scores[currind]
            vM[l] = RU
            rM[l] = currind
            
            Un[l,l] = -1
            Un[:l,l] = Un[:l,:l] @ Undn[currind,:l].T
            Un[:,l] = - Un[:,l] / np.sqrt(norm2[currind])
            if l == N0-1 :
                break
            Gs[l] = G[currind]
            Undn[:,l] = Un[:l+1,l].T.conj() @ Gs[:l+1,:]
            
            Rdn = Rdn - vM[l] * Undn[:,l].conj()
            norm2 = norm2 - np.abs(Undn[:,l])**2
            norm2[norm2<0] = 0 # sometimes happens to have small negative numbers
            non_null = (norm2 > 1e-8)
            scores = np.zeros(J,dtype=complex)
            scores[non_null] = Rdn[non_null] / np.sqrt(norm2)[non_null]

        vM = Un @ vM
        
        indices = [j for j in rM if j >= 0]
        A_c[indices,k] = vM[:len(indices)]
            
#    elapsed = (time.time() - start)
#    print('time : ',elapsed)

    return A_c



'''ORMP_cholesky_real() and OMP() are not used in this script, but it is interesting to
write them as a comparison to ORMP_cholesky(). Empirically, OMP() behaves badly compared
ORMP_cholesky_real() for small values of N0, and becomes better after increasing N0.
ORMP_cholesky_real() is less performing than ORMP_cholesky() because the former
can only rely on real weights in sparse coding, contrarily to the latter. '''

# NOT USED HERE
def ORMP_cholesky_real(DM,N0,J,dataset) :
    ''' Same as ORMP_cholesky but in the real setting.
    It copies the function of Mairal et al. (2009) for the SPAMS toolbox, as a comparison. '''
    sqrtPsi = scipy.linalg.sqrtm(Psi).real
    import spams
    K = len(dataset)
    dataset_r = multi_complex2real(dataset)
    DR = np.asfortranarray(sqrtPsi @ multi_complex2real(DM.T).T)
    X = np.asfortranarray(sqrtPsi @ dataset_r.T)
    AM = spams.omp(X,D=DR,L=N0)
    AM = np.array(AM.todense()).reshape(J,K)
    return AM


# NOT USED HERE
def matPlus(B) : 
    '''Auxiliary function of OMP().
    Gives the pseudo-inverse   (B^H @ Phi @ B)^{-1} @ B^H @ Phi   of a matrix B
    with respect to the Hermitian product Phi.'''
    try :
        Bp = np.linalg.inv(B.T.conj() @ Phi @ B) @ B.T.conj() @ Phi
    except np.linalg.LinAlgError :
        global prob
        prob = B.T.conj() @ Phi @ B
        print('not invertible !!!\n',prob)
    else :
        return Bp
    
# NOT USED HERE
def OMP(D_c,dataset,N0) : # columns of D_c must be preshapes
    ''' Complex Orthogonal Matching Pursuit (complex OMP version).
    Requires that the atoms contained in D_c are all preshapes.
    '''
    print('Caution! this is a naive implementation that does not handle singular cases.')
    J = D_c.shape[1]
    K = len(dataset)
    A_c = np.zeros((J,K),dtype = complex)    

    for k in range(K) :
        z = dataset[k]
        rem = z
        indices = np.array([],dtype = int)    
        for i in range(N0) :
            corr = D_c.T.conj() @ Phi @ rem
            corr[indices] = 0 # mathematically should be zero
            j0 = int(np.argmax(np.abs(corr)))
            indices = np.append(indices,j0)
            DD_I = D_c[:,indices]
            DD_I_plus = matPlus(DD_I)
            P = DD_I @ DD_I_plus
            rem = z - P @ z
        a_I = DD_I_plus @ z # in C^N0
        A_c[indices,k] = a_I
    return A_c


''' PRESHAPING THE ATOMS OF D_c '''

# NOT USED HERE
def proj_S(D_c) : # columns of D_c are general configurations in C^N
    '''converts non-zero columns of D_c to preshapes, by projecting them on the subspace
    rthogonal to u       {d | u.T.conj() Phi d = 0)}
    and then normalizing them. Those which are zero are left as they are.
    '''
    from settings import uu
    J = D_c.shape[1]

    for j in range(J) :
        d = D_c[:,j]
        no = norm(d)
        if no > 1e-8 :
            orth_d = her(uu,d)*uu
            d = d - orth_d
            d = d / norm(d)
            D_c[:,j] = d
    return D_c

def normalize(D_c) : # columns of D_c are centered configurations
    ''' Same as proj_S(), but we know that all the columns are already centered. '''
    J = D_c.shape[1]
    
    for j in range(J) :
        d = D_c[:,j]
        no = norm(d)
        if no > 1e-8 :
            D_c[:,j] = d/no
    return D_c


''' DISPLAY FUNCTIONS '''

from settings import z0
    

def display_res(dataset,DA,k,save = False,directory = None) :
    '''Auxiliary function of display. display_res() show the reconstruction for data z_k'''
    print('k = ',k)
    z = dataset[k]
    w = DA[:,k] # not necessarily normalized
    ta = theta(z,z0) ; z = np.exp(1j*ta)*z ; w = np.exp(1j*ta)*w
    norm_error = np.round(norm(z - w).real,4) 
    draw(z,i = 7,comparing = True) # i = 7 for gray
    draw(w,i = 0)
    plt.axis('equal')
    plt.axis('off')
    title_text = '$ |z_k - D \\alpha_k|_\\Phi : {} \% $'.format(np.round(100*norm_error,4))
    plt.title(title_text, fontsize = 20, y = 1)
    if save :
        plt.savefig(directory + '/rec_'+str(k)+'.png',dpi = 200)
    plt.show()


def display(D_c,A_c,dataset,test_k_set=test_k_set,save = False,directory = None) :
    '''Displays some reconstructions, and optionally the sparse
    coefficients A.'''
    
    DA = D_c @ A_c

    for k in test_k_set :
        display_res(dataset,DA,k,save = save,directory = directory)



''' MAIN ALGORITHM : 2D Kendall Shape Dictionary with the Method of Optimal Directions and 
    Cholesky-optimized Order Recursive Matching Pursuit '''


def reciprocal(sigmas):
    '''Auxiliary function of KSD_optimal_directions().
    Given a 1D array of non-negative elements called sigmas, with possibly zero elements,
    returns the array of multiplicative inverses whenever possible, and leaves the zeroes.'''
    sigmas_rec = np.zeros_like(sigmas)
    for i,x in enumerate(sigmas) :
        if x != 0 :
            sigmas_rec[i] = 1/x
    return sigmas_rec

def fill_diagonal(sigmas_rec,J,K):
    '''Auxiliary function of KSD_optimal_directions().
    Fills in the diagonal of a matrix of shape (K,J).'''
    Sigma_rec = np.zeros((K,J))
    np.fill_diagonal(Sigma_rec,sigmas_rec)
    return Sigma_rec


def KSD_optimal_directions(dataset,N0,J,init=None,Ntimes = 100,verbose=False,save=False,directory = None) :
    ''' The 2D Kendall Shape Dictionary classically alternates between:
    
    - a sparse coding step : the weights A are updated using a Cholesky-based 
    Order Recursive Matching Pursuit (ORMP), as a direct adaptation to the
    complex setting of Mairal's implementation for the real setting in the SPAMS toolbox.
    - a dictionary update : following the Method of Optimal Directions (MOD),
    we update D as
    
            D <- [z_1,...,z_K] @ A^H @ (A @ A^H)^{-1}
            D <- Pi_S(D) (center and normalize all the non-null atoms d_j)
    
    and then replace under-utilized or null atoms by randomly picked data.
    An atom d_j is arbitrarily said to be under-utilized if 
            (nb of data using d_j) / (K*N0) < 1 / (50*J)
            
            
    Parameters:
        - dataset in C^{(K,n)} is a complex array containing the horizontally stacked dataset [z_1,...,z_K]^T
        - N0 determines the L0 sparsity of the weights a_k
        - J fixes the number of atoms that we want to learn
        - init = None initializes the dictionary with randomly picked data shapes.
            if init is a given (n,J) complex array, then the initialization starts with init.
        - Ntimes is the number of iterations
        - if verbose == True, the algorithm keeps track of the loss function E to be minimized at each iteration.
            It saves time to set verbose = False.
        
    '''
    K = len(dataset)    
    if type(init) == np.ndarray :
        D_c = init
    else :
        D_c = initializeD_c(J,dataset)
    if verbose :
        print('Initializing the dictionary.')
        drawMany(D_c.T,force = True)
        lossCurve = np.array([])
    
    print("Let's wait for {} iterations...".format(Ntimes))
    start = time.time()

    for t in range(Ntimes) :
        
        if t % 5 == 0 :
            print('t =',t)
            
        A_c = ORMP_cholesky(D_c,dataset,N0)
        
        if verbose :
            diffs = dataset.T - D_c @ A_c
            E = np.diag(diffs.T.conj() @ Phi @ diffs).sum().real
            lossCurve = np.append(lossCurve,E)
            
        try :
            Mat = np.linalg.inv(A_c @ A_c.T.conj())
        except np.linalg.LinAlgError :
            global A_error
            A_error = A_c
            print('A @ A^H not invertible, using SVD')
            U,sigmas,VH = np.linalg.svd(A_c)
            sigmas_rec = reciprocal(sigmas)
            Sigma_rec = fill_diagonal(sigmas_rec,J,K)
            D_c = dataset.T @ VH.T.conj() @ Sigma_rec @ U.T.conj()
        else :
            D_c = dataset.T @ A_c.T.conj() @ Mat

        D_c = normalize(D_c) # the new atoms are preshaped

        purge_j = np.where((np.abs(A_c)>1e-3).sum(axis=1)/K < N0/(5*J))[0]
        for j in range(J) :
            if norm(D_c[:,j]) < 1e-8 or j in purge_j :
                print('purged ',j,'at iteration',t)
                D_c[:,j] = shapes[np.random.randint(K)]

    print('computing the final weights...')
    A_c = ORMP_cholesky(D_c,dataset,N0) ; 
                
    elapsed = (time.time() - start)
    print('duration of the algorithm: ', np.round(elapsed,2), 'seconds')
    
    diffs = dataset.T - D_c @ A_c
    E = np.diag(diffs.T.conj() @ Phi @ diffs).sum().real
    
    print('FINAL RESULTS')
    display(D_c,A_c,dataset, save = save,directory=directory) 
    
    if verbose :
        lossCurve = np.append(lossCurve,E)
        plt.figure() ;
        plt.plot(np.arange(len(lossCurve)),lossCurve)
        plt.title('Loss curve for the KSD algorithm')
        if save : plt.savefig(directory + '/losscurve.png',dpi = 100)
    plt.show()
        
    drawMany(D_c.T,force = True,show = False)
    plt.title('KSD dictionary  N0 = {}  J = {}'.format(N0,J))
    if save :
        plt.savefig(directory + '/dico_KSD.png',dpi = 200)
    plt.show()

    D_al = align_rot(D_c.T).T
    drawMany(D_al.T,force = True,show = False)
    plt.title('KSD rotated dictionary  N0 = {}  J = {}'.format(N0,J))
    if save :
        plt.savefig(directory + '/dico_KSD_rotated.png',dpi = 200)
    plt.show()

    print('Final loss : ',E)
    if save :
        text_file = open(directory + '/readme.txt','a')
        text_file.write('\nduration of the algorithm: {} s \n \n'.format(np.round(elapsed,2)))
        text_file.write('Final loss: {}\n'.format(E))
        text_file.close()

    return D_c,A_c,E


if __name__ == '__main__' :
    
    learn_dico = True
    N0,J = 6,20
    SAVE = True
    
    if learn_dico :
        
        directory = 'RESULTS/' + database[choice] + '/N0_'+ str(N0) + '_J_' + str(J) 

        if SAVE :
            if not os.path.exists(directory):
                print('CREATING THE FOLDER')
                os.makedirs(directory)
            else :
                if os.path.exists(directory + '/dico_KSD.png') :
                    SAVE = False
                    print('WILL NOT OVERWRITE PREVIOUS SAVED RESULTS')
    
                
        D_c,A_c,E_KSD = KSD_optimal_directions(shapes,N0,J,init = None,
                                           Ntimes = 40,verbose=True,
                                           save=SAVE,directory=directory)
            
            
        
   