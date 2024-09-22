import numpy as np
from .mps import MPS
from .mpo import MPO
from .krylov import eigh_krylov
from .hamiltonian import molecular_hamiltonian_mpo
import copy
from scipy import sparse
import pickle

def get_elementary_mpo_qn(upper_index, op_value,  X_mo, L):
    
    #set size for "simple" MPO
    qd = np.array([0, 1])
    #qD = [np.array([0]), np.array([0, -1]) for i in range(L-1), np.array([-1])]
    #Pauli matrices:
    I2 = np.identity(2)
    bose_c = np.array([[0, 0.],[1, 0]])
    bose_a = np.array([[0, 1.],[0, 0]])
    pauli_Z = np.array([[1, 0.],[0, -1]])
    
    #annihilation
    if op_value == 0:
        
        qD = [np.array([0])] + [np.array([0, -1]) for i in range(L-1)] + [np.array([-1])]
        mpo = MPO(qd, qD, fill= 0.)

        mpo.A[0][:, :, 0, 0] = pauli_Z
        mpo.A[0][:, :, 0, 1] = bose_a * X_mo[upper_index, 0]

        mpo.A[L-1][:, :, 0, 0] = bose_a * X_mo[upper_index, L-1]
        mpo.A[L-1][:, :, 1, 0] = I2 

        for i in range (1,L-1):
            mpo.A[i][:, :, 0, 0] = pauli_Z
            mpo.A[i][:, :, 0, 1] = bose_a * X_mo[upper_index, i]
            mpo.A[i][:, :, 1, 0] = 0
            mpo.A[i][:, :, 1, 1] = I2

    #creation
    if op_value == 1:
        
        qD = [np.array([0])] + [np.array([0, 1]) for i in range(L-1)] + [np.array([1])]
        mpo = MPO(qd, qD, fill= 0.)
        
        mpo.A[0][:, :, 0, 0] = pauli_Z
        mpo.A[0][:, :, 0, 1] = bose_c * X_mo[upper_index, 0]

        mpo.A[L-1][:, :, 0, 0] = bose_c * X_mo[upper_index, L-1]
        mpo.A[L-1][:, :, 1, 0] = I2 

        for i in range (1,L-1):
            mpo.A[i][:, :, 0, 0] = pauli_Z
            mpo.A[i][:, :, 0, 1] = bose_c * X_mo[upper_index, i]
            mpo.A[i][:, :, 1, 0] = 0
            mpo.A[i][:, :, 1, 1] = I2

    return mpo
