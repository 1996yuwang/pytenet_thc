import numpy as np
from pytenet.mps import MPS
import copy


def create_0_mps(bond_set, dir):
    ''' 
    bond_set: bond dims
    create a MPS with all entries 0
    '''
    d = 2
    D = bond_set
    qd = np.array([0, 1])
    qD = [np.zeros(Di, dtype=int) for Di in D]
    A = MPS(qd, qD, fill=0.)
    A.orthonormalize(mode = dir)

    return A

def generate_single_state(L, state):
    '''  
    state: a single computational basis e.g., |1,1,1,1,0,0,0,0,0,0>
    L: MPS nsites
    state: state vector written as computational basis, written as list.
    Given a state vector e.g., |1110010101>, generate its mps
    MPS qnumber direction:
          |
          |
          V
    --->  O --->
    '''
    assert L == len(state)
    bond_set_1 = [1] * (L+1)
    mps_single = create_0_mps(bond_set_1, 'left')
    n_electron = np.sum(state)
    #mps_single.qD[L+1] = copy.deepcopy(n_electron)
    n_accumulation = 0
    mps_single.qD[0] = np.array([n_accumulation])
    for i in range (L):

        if state[i] == 0:
            (mps_single.A[i])[1] = 0
            (mps_single.A[i])[0] = 1
            mps_single.qD[i+1] = np.array([n_accumulation])
            
        if state[i] == 1:
            mps_single.qD[i+1] = np.array([1])
            (mps_single.A[i])[1] = 1
            (mps_single.A[i])[0] = 0
            n_accumulation += 1
            mps_single.qD[i+1] = np.array([n_accumulation])
            
    assert n_accumulation == n_electron
    
    return mps_single