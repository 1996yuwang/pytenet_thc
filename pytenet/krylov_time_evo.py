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
import copy
from pytenet.operation_thc import apply_thc_mpo_and_compress, add_mps_and_compress
from pytenet.operation import vdot, add_mps, operator_inner_product




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
    
    #print(c_reduced)

    psi_evloved = np.zeros_like(psi, dtype=np.complex128)

    for i in range (len(Krylov_vector_list)):
        psi_evloved += c_reduced[i] * Krylov_vector_list[i]
        
    return psi_evloved

def Krylov_time_evo_using_vecs(H, N_krylov, psi, n, T):
    
    dt = T/n 
    for i in range (n):
        psi = Krylov_evo_using_vecs_single_step(H, N_krylov, psi, dt)
        
    return psi

def create_Krylov_space(N_Krylov, H_mu_nu_list_spin_layer, last_mps, trunc_tol, max_bond, r_THC):
    
    Krylov_space = []
    
    v0 = copy.deepcopy(last_mps)
    v0.orthonormalize('left')
    v0.orthonormalize('right')
    Krylov_space.append(v0)
    #print(v0.bond_dims)
    
    v1 = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(v0), trunc_tol, 2*max_bond, r_THC)
    #v1 = apply_operator(H_mu_nu_list_spin_layer, copy.deepcopy(last_mps), trunc_tol, max_bond, r_THC)
    #v1.orthonormalize('left')
    #v1.orthonormalize('right')
    temp = copy.deepcopy(v0)
    temp.A[0] = -vdot(v1, temp)* temp.A[0]
    v1 =  add_mps_and_compress(copy.deepcopy(v1), temp, trunc_tol, max_bond)
    v1.orthonormalize('left')
    v1.orthonormalize('right')
    Krylov_space.append(v1)
    #print(v1.bond_dims)
    
    for i in range(2, N_Krylov, 1):
        
        v_i = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(Krylov_space[i-1]), trunc_tol, 2*max_bond, r_THC)
        v_i.orthonormalize('left')
        v_i.orthonormalize('right')
        
        temp1 = copy.deepcopy(Krylov_space[i-2])
        temp1.A[0] = -vdot(v_i, temp1)* temp1.A[0]
        v_i =  add_mps(copy.deepcopy(v_i), temp1)
        #v_i.orthonormalize('left')
        #v_i.orthonormalize('right')
        
        temp2 = copy.deepcopy(Krylov_space[i-1])
        temp2.A[0] = -vdot(v_i, temp2)* temp2.A[0]
        v_i =  add_mps_and_compress(copy.deepcopy(v_i), temp2, trunc_tol, max_bond)
        v_i.orthonormalize('left')
        v_i.orthonormalize('right')
        #print(v_i.bond_dims)
        Krylov_space.append(v_i)
        
        # print(vdot(Krylov_space[i], Krylov_space[i-1]))
        # print(vdot(Krylov_space[i], Krylov_space[i-2]))
        
    return Krylov_space

def Krylov_evo_using_built_mps_space(H_mpo, Krylov_mps_list, max_bond, dt):
    
    TN = np.zeros([len(Krylov_mps_list),len(Krylov_mps_list)])
    for i in range (TN.shape[0]):
        for j in range (TN.shape[1]):
            if abs(i - j) < 2:
                TN[i, j] = operator_inner_product(Krylov_mps_list[i], H_mpo, Krylov_mps_list[j])
                
    c1 = np.zeros([len(Krylov_mps_list), 1])
    c1[0,0] = 1
    exp_TN = spla.expm(-1j*dt*TN)
    c_reduced = exp_TN@ c1

    psi_evloved = copy.deepcopy(Krylov_mps_list[0])
    psi_evloved.A[0] = c_reduced[0] *psi_evloved.A[0]

    for i in range (1, len(Krylov_mps_list), 1):
        temp = copy.deepcopy(Krylov_mps_list[i])
        temp.A[0] = c_reduced[i] *temp.A[0]
        psi_evloved = add_mps(psi_evloved, temp)
    
    psi_evloved.orthonormalize('right')
    psi_evloved.orthonormalize('left')
        
    psi_evloved.compress_direct_svd_right_max_bond(0, max_bond)
    psi_evloved.orthonormalize('right')
    psi_evloved.orthonormalize('left')
        
    return psi_evloved  


    
def Krylov_evo_using_built_space(H, Krylov_vector_list, dt):
    
    TN = np.zeros([len(Krylov_vector_list),len(Krylov_vector_list)])
    for i in range (TN.shape[0]):
        for j in range (TN.shape[1]):
            if abs(i - j) < 2:
                TN[i, j] = np.vdot(Krylov_vector_list[i], H @Krylov_vector_list[j])
                
    c1 = np.zeros([len(Krylov_vector_list), 1])
    c1[0,0] = 1
    exp_TN = spla.expm(-1j*dt*TN)
    c_reduced = exp_TN@ c1

    psi_evloved = np.zeros_like(Krylov_vector_list[0], dtype=np.complex128)

    for i in range (len(Krylov_vector_list)):
        psi_evloved += c_reduced[i] * Krylov_vector_list[i]
    
    psi_evloved /= norm(psi_evloved)
        
    return psi_evloved   

