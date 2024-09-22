import numpy as np
from pytenet.mps import MPS


def create_0_mps(bond_set, dir):
    ''' 
    bond_set: bond dims
    create a MPS with all entries 0
    '''
    d = 2
    D = bond_set
    qd = np.zeros(d, dtype=int)
    qD = [np.zeros(Di, dtype=int) for Di in D]

    A = MPS(qd, qD, fill=0.)
    A.orthonormalize(mode = dir)
 
    return A

def generate_single_state(L, state):
    '''  
    L: MPS nsites
    state: state vector written as computational basis, written as list.
    Given a state vector e.g., |1110010101>, generate its mps
    '''
    assert L == len(state)
    bond_set_1 = [1] * (L+1)
    mps_single = create_0_mps(bond_set_1, 'left')

    for i in range (L):
        if state[i] == 1:
            (mps_single.A[i])[1] = 1
            (mps_single.A[i])[0] = 0
        if state[i] == 0:
            (mps_single.A[i])[1] = 0
            (mps_single.A[i])[0] = 1
    
    return mps_single