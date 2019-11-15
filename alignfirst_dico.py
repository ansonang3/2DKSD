#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anna SONG

alignfirst_dico.py defines the `naive' version of dictionary learning closest to
our problem. It consists in first optimally rotating any data to a common reference
configuration (e.g. the mean of the dataset), and then applying a standard
dictionary learning on the rotated dataset.

All this is done in the REAL framework. That's why the real scalar product Psi is used here,
not the Hermitian product Phi.

Relies on the SPAMS toolbox of Mairal et al. (2009).


Warning:
    
    Mathematical shapes can only be numerically handled as preshapes.
    
    Hence the `shape' variable designates in fact a preshape
    (centered and normalized configuration), that is one of the many representatives
    of the equivalence class defining the corresponding (mathematical) shape.

"""

import numpy as np
import matplotlib.pyplot as plt

from settings import database, choice, K
from settings import shapes, multi_complex2real, multi_real2complex, Psi, Phi
from settings import draw, drawMany, test_k_set, norm, z0

from shapespace import align_rot, theta

import spams
import scipy
import os

sqrtPhi = scipy.linalg.sqrtm(Phi)
sqrtPhi_inv = np.linalg.inv(sqrtPhi)
sqrtPsi = scipy.linalg.sqrtm(Psi).real
sqrtPsi_inv = np.linalg.inv(sqrtPsi).real


''' DISPLAY FUNCTION '''

def display_res(dataset,DA,k,save = False,directory = None) :
    '''display_res() show the reconstruction for data z_k'''
    print('k =',str(k))
    z = dataset[k]
    w = DA[:,k] # not necessarily normalized
    ta = theta(z,z0) ; z = np.exp(1j*ta)*z ; w = np.exp(1j*ta)*w
    norm_error = np.round(norm(z - w).real,4) # in the ambient space C^N
    draw(z,i = 7,comparing = True) # i = 7 for gray
    draw(w,i = 0)
    plt.axis('equal')
    plt.axis('off')
    title_text = '$ |z_k - D \\alpha_k|_\\Phi : {} \% $'.format(np.round(100*norm_error,4))
    plt.title(title_text, fontsize = 20, y = 1)

    if save :
        plt.savefig(directory + '/rec_'+ str(k) + '_alignfirst.png',dpi = 200)
    plt.show()
    

''' ALIGN-FIRST METHOD (using SPAMS, real setting) '''

def alignfirst_dico(dataset,N0,J,init = None,save = False,directory=None,verbose=False) :
    '''Performs (real) dictionary learning on the dataset, after it is optimally rotated along its mean.
    Relies on the SPAMS toolbox of Mairal et al. '''
    K1 = len(dataset)
    dataset = align_rot(dataset)
    dataset_r = multi_complex2real(dataset)
    X = sqrtPsi @ dataset_r.T
    X = np.asfortranarray(X) # necessary for using spams toolbox
    D = spams.trainDL(X,K = J,D = init,mode=3,modeD=0,lambda1 = N0,verbose=verbose)
    A = spams.omp(X,D=D,L=N0)
    Ad = np.array(A.todense()).reshape(J,K)
    D_c = multi_real2complex((sqrtPsi_inv @ D).T).T
    
    drawMany(D_c.T,show = False)
    plt.title('Align-first dictionary  N0 = {}  J = {}'.format(N0,J))
    if save :
        plt.savefig(directory + '/dico_alignfirst.png',dpi = 200)
    plt.show()
    
    if verbose :  
        DA = D_c @ A
        for k in test_k_set :
            display_res(dataset,DA,k,save = save,directory = directory)
        
    diffs = dataset.T - D_c @ Ad
    if K1 < 10000 :
        E = np.diag(diffs.T.conj() @ Phi @ diffs).sum().real
    else :
        E = 0
        for k in range(K) :
            E += (diffs[:,k].conj() @ Phi @ diffs[:,k]).real
    print('final loss : ',E)
    print('RMSE :',np.sqrt(E/K))
    if save :
        text_file = open(directory + '/readme_alignfirst.txt','a')
        text_file.write('Final loss: {}\n'.format(E))
        text_file.write('Final RMSE: {}\n'.format(np.sqrt(E/K)))
        text_file.close()
    return D_c, Ad, E


def alignfirst_RMSE_curve(dataset,J) :
    '''
    For a fixed J (nb of atoms imposed), for any N0 <= J,
    run align-first using SPAMS and record the RMSE(N0,J). Return the RMSE curve.
    
    Do not run on too large datasets!! (K > 10000)
    '''
    K = len(dataset)
    curve = np.zeros(J)
    for N0 in range(1,J+1) :
        D,A,E_AF = alignfirst_dico(shapes,N0,J,verbose=False)
        curve[N0-1] = np.sqrt(E_AF/K)
    plt.plot(curve)
    plt.show()
    
    return curve


''' COMPLEX PCA method '''

def complex_PCA(dataset,J) :
    '''
    Computes the first J modes of complex PCA and the RMSE rate of the N0-term reconstruction,
    for 1 <= N0 <= J.
    '''
    K = len(dataset)
    
    average = np.mean(dataset,axis = 0)
    dataset_a = dataset - average
    
    sum_norms = np.diag(dataset_a.conj() @ Phi @ dataset_a.T).sum().real
    
    Z = dataset_a.T # dataset in columns
    Y = Z.T.conj() @ sqrtPhi # deforming through sqrtPhi to recast to a standard form
    Lambdas, Vmodes = np.linalg.eig(Y.T.conj() @ Y) # complex values!
    Lambdas = Lambdas.real
    sorting = np.argsort(Lambdas)
    sorting = sorting[::-1] # eigenvalues ordered by decreasing absolute value
    Lambdas = Lambdas[sorting[:J]]
    Vmodes = Vmodes[:,sorting[:J]]
    Atoms = np.linalg.inv(sqrtPhi) @ Vmodes # recasting to our form
    
    cumul = np.cumsum(Lambdas)
    E_rate = sum_norms - cumul
    
    RMSE_rate = np.sqrt(E_rate/K)
    plt.plot(np.arange(J),RMSE_rate)
    
    drawMany(Atoms.T,force = True,show=False)
    plt.title('PCA modes J = {}'.format(J))
    plt.show()
    
    Recs = Atoms @ (Atoms.T.conj() @ Phi @ Z)
    Recs = (Recs.T + average).T
    
    for k in test_k_set :
        display_res(dataset,Recs,k)

    return RMSE_rate


def display_rates(PCA_RMSE,AF_RMSE,KSD_RMSE,J,filename='RMSE_rates'):
    plt.figure() ; ax = plt.gca()
    xaxis = np.arange(1,J+1)
    ax.set_xticks(np.arange(1,J+1))
    ax.plot(xaxis,PCA_RMSE,marker = 'o')
    ax.plot(xaxis,AF_RMSE, marker = '^')
    ax.plot(xaxis,KSD_RMSE, marker = '*')
    ax.set_yscale('log')
    ax.set_xlabel('N0')
    ax.set_ylabel('RMSE')
    #ax.set_ylim(top=None,bottom=None)
    ax.legend(['PCA','A-F','KSD'])
    plt.savefig(filename,dpi=200)
    plt.show()


if __name__ == '__main__' :
    aligned = align_rot(shapes)
    print('Aligning the shapes...')
    drawMany(aligned)

    learn_dico = True # if True then launches dictionary learning
    N0,J = 5,10
    # N0: nb of atoms to be picked to reconstruct (l0 penalty)
    # J: number of atoms to learn
    SAVE = True # do we save the results into the directory?
    
    if learn_dico :
        
        if SAVE :
            directory = 'RESULTS/' + database[choice] + '/N0_'+ str(N0) + '_J_' + str(J)

            if not os.path.exists(directory):
                print('CREATING THE FOLDER')
                os.makedirs(directory)
            else :
                if os.path.exists(directory + '/dico_alignfirst.png') :
                    SAVE = False
                    print('WILL NOT OVERWRITE PREVIOUS SAVED RESULTS')

        D,A,E_AF = alignfirst_dico(shapes,N0,J,save = SAVE,directory=directory,verbose=True)
        
        for j in range(J) : # normalize to obtain preshaped atoms (the atoms are already centered)
            no = norm(D[:,j])
            if no > 1e-3 :
                D[:,j] /= no
                A[:,] *= no
            else :
                print('did not normalize because small norm for j = ',j)
