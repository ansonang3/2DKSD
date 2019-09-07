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

from settings import database, choice
from settings import shapes, multi_complex2real, multi_real2complex, Psi, Phi
from settings import draw, drawMany, test_k_set, norm, z0

from shapespace import align_rot, theta

import spams
import scipy
import os

sqrtPsi = scipy.linalg.sqrtm(Psi).real
sqrtPsi_inv = np.linalg.inv(sqrtPsi).real

    
''' ALIGN-FIRST METHOD (using SPAMS, real setting) '''

def alignfirst_dico(dataset,N0,J,init = None,save = False,directory=None) :
    '''Performs (real) dictionary learning on the dataset, after it is optimally rotated along its mean.
    Relies on the SPAMS toolbox of Mairal et al. '''
    K = len(dataset)
    dataset = align_rot(dataset)
    dataset_r = multi_complex2real(dataset)
    X = sqrtPsi @ dataset_r.T
    X = np.asfortranarray(X) # necessary for using spams toolbox
    D = spams.trainDL(X,K = J,D = init,mode=3,modeD=0,lambda1 = N0,verbose=False)
    A = spams.omp(X,D=D,L=N0)
    Ad = np.array(A.todense()).reshape(J,K)
    D_c = multi_real2complex((sqrtPsi_inv @ D).T).T
    
    drawMany(D_c.T,show = False)
    plt.title('Align-first dictionary  N0 = {}  J = {}'.format(N0,J))
    if save :
        plt.savefig(directory + '/dico_alignfirst.png',dpi = 200)
    plt.show()
    
    DA = D_c @ A
    
    for k in test_k_set :
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
        
    diffs = dataset.T - D_c @ Ad
    E = np.diag(diffs.T.conj() @ Phi @ diffs).sum().real
    print('final loss : ',E)
    if save :
        text_file = open(directory + '/readme_alignfirst.txt','a')
        text_file.write('Final loss: {}\n'.format(E))
        text_file.close()
    return D_c, Ad, E



if __name__ == '__main__' :
    aligned = align_rot(shapes)
    print('Aligning the shapes...')
    drawMany(aligned)

    learn_dico = True # if True then launches dictionary learning
    N0,J = 6,20
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

        D,A,E_AF = alignfirst_dico(shapes,N0,J,save = SAVE,directory=directory)
        
        for j in range(J) : # normalize to obtain preshaped atoms (the atoms are already centered)
            no = norm(D[:,j])
            D[:,j] /= no
            A[:,] *= no
