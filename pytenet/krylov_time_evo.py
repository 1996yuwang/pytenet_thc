#framework:
'''
def Krylov_time_evo_mps(N_space, time_step_size, t, thc_mpo, initial_mps, trunc_tol, max_bond_dim, H_ref_thc, H_ref)
while calculate errors, compare:
ED vs ED using H_ref_thc (thc error)
ED vs Krylov using H_ref (Krylov error)
Krylov using H_ref_thc using exact vectors vs Krylov using MPS using thc mpo (MPS error)

def Krylov_time_evo_single_step_mps(N_space, time_step_size, thc_mpo, last_mps, trunc_tol, max_bond_dim, H_ref_thc, H_ref)
implement single step time-evo according to these inputs, 
while calculate errors, compare:
ED vs ED using H_ref_thc (thc error)
ED vs Krylov using H_ref (Krylov error)
Krylov using H_ref_thc using exact vectors vs Krylov using MPS using thc mpo (MPS error)

def create_Krylov_space(N_space, thc_mpo, last_mps, trunc_tol, max_bond_dim)
create Krylov subspace according to these inputs.

def time_evo_using_Krylov_vectors
given Krylov subspace created by last func, do time-evo

def estimate_thc_error(H_ref, H_thc, initial, T)  
given total time and Hamiltonians, calculate thc error brought to time-evo
please use ED_time_evo(H, initial, dt, T)

def estimate_krylov_error(H_ref, initial, T)
given total time and Hamiltonian, calculate krylov error brought to time-evo
please use ED_time_evo(H, initial, dt, T) and Krylov_time_evo_using_vecs(H, initial, dt, T)

def estimate_trunc_error() 
please use Krylov_time_evo_mps and Krylov_time_evo_using_vecs(H, initial, dt, T)

def ED_time_evo(H, initial, dt, T)
calculate time-evolved vectors using ED, with H_ref or H_thc

def Krylov_time_evo_using_vecs(H, N_krylov, initial, n, T)
calculate time-evolved vectors using Krylov, with H_ref or H_thc

def Krylov_time_evo_using_vecs_single_step(H, N_krylov, initial, dt)
'''

import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as spla



def gram_schmidt(vectors):
    #gram-schmidt a group of vectors
    orthogonal_vectors = []
    
    for v in vectors:
        # Start with the vector v
        w = v.copy()
        
        # Subtract the projections of v onto each of the previous orthogonal vectors
        for u in orthogonal_vectors:
            projection = np.dot(v, u) / np.dot(u, u) * u
            w -= projection
        
        # Append the orthogonalized vector w
        orthogonal_vectors.append(w)
    
    return orthogonal_vectors

def ED_time_evo(H, psi, T):
    return(spla.expm_multiply(-1j* (T)* H, psi))

def Krylov_evo_using_vecs_single_step(H, N_krylov, psi, dt):
    
    #create krylov space:

    Krylov_vector_list = []
    
    psi /= norm(psi)
    Krylov_vector_list.append(psi)

    for i in range (N_krylov-1):
        krylov_vector = H@ Krylov_vector_list[i]
        krylov_vector /= norm(krylov_vector)
            
        for j in range (i+1):    
            krylov_vector +=  -np.vdot(Krylov_vector_list[j], krylov_vector)*Krylov_vector_list[j]
        
        krylov_vector /= norm(krylov_vector)
        Krylov_vector_list.append(krylov_vector)
    
    #test the orthogonality
    # for i in range (len(Krylov_vector_list)):
    #     for j in range (len(Krylov_vector_list)):
    #         if i != j:
    #             assert abs(np.vdot(Krylov_vector_list[i], Krylov_vector_list[j])) < 1e-10
    
    #reduced Hmailtonian:    
    TN = np.zeros([len(Krylov_vector_list),len(Krylov_vector_list)])
    for i in range (TN.shape[0]):
        for j in range (TN.shape[1]):
            if abs(i - j) < 2:
                TN[i, j] = np.vdot(Krylov_vector_list[i], H@Krylov_vector_list[j])
    
        
    c1 = np.zeros([len(Krylov_vector_list), 1])
    c1[0,0] = 1
    exp_TN = spla.expm(-1j*dt*TN)
    c_reduced = exp_TN@ c1

    psi_evloved = np.zeros_like(psi, dtype=np.complex128)

    for i in range (len(Krylov_vector_list)):
        psi_evloved += c_reduced[i] * Krylov_vector_list[i]
        
    return psi_evloved

def Krylov_time_evo_using_vecs(H, N_krylov, psi, n, T):
    
    dt = T/n 
    for i in range (n):
        psi = Krylov_evo_using_vecs_single_step(H, N_krylov, psi, dt)
        
    return psi
        