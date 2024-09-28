import numpy as np
import pickle
import copy
from .operation import vdot
from .operation_thc import apply_thc_mpo_and_compress, add_mps_and_compress
from scipy import sparse

__all__ = ['generate_krylov_space_in_disk', 'ortho_to_previous_two', 'get_W', 'get_S', 'ortho_to_previous_two', 'generate_re_ortho_space', 'generate_reduced_H', 'generate_Hamiltonian_with_occupation_number']

def generate_krylov_space_in_disk(N_Krylov, H_mu_nu_list_spin_layer, psi_original, max_bond_Krylov, trunc_tol, r_THC, foldername):  
    '''  
    generate orthogonalized Krylov space, and store in disk.
    
    N_Krylov: the size of Krylov space.
    
    H_mu_nu_list_spin_layer: THC-MPO, as a list of MPOs.
    
    psi_original: initial state v_0 for Krylov methods.
    
    max_bond_Krylov: maximum bond dimension for Krylov vectors.
    
    foldername: one must input a foldername where the Krylov vectors are stored in.
    '''
    
    # store the v_0 in disk
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(copy.deepcopy(psi_original), file)

    # H v_0 and orthogonalize it, then store in disk.
    H_on_psi = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(psi_original), trunc_tol, max_bond_Krylov, r_THC)
    H_on_psi.orthonormalize('right')
    print(H_on_psi.bond_dims)

    temp = copy.deepcopy(psi_original)
    temp.A[0] = -vdot(H_on_psi, temp)* temp.A[0]
    H_on_psi =  add_mps_and_compress(copy.deepcopy(H_on_psi), temp, trunc_tol, max_bond_Krylov)
    H_on_psi.orthonormalize('right')

    filename = foldername + f"/Krylov_vec{1}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(H_on_psi, file)

    # from now on, all the H v_i should be orthogonalized to the previous two vectors.
    for i in range (2, N_Krylov):
        
        #show how many Krylov vectors are achieved     
        print(i)    
        # if i % 5 == 0:
        #     print("implemented:", i)
        
        #from disk load the previous two Krylov vectors.    
        filename = foldername + f"/Krylov_vec{i-2}.pkl"
        with open(filename, 'rb') as file:
            orth_state1 = pickle.load(file)
        filename = foldername + f"/Krylov_vec{i-1}.pkl"
        with open(filename, 'rb') as file:
            orth_state2 = pickle.load(file)
        
        #first calculate H \v_i
        this_state = apply_thc_mpo_and_compress(H_mu_nu_list_spin_layer, copy.deepcopy(orth_state2), trunc_tol, max_bond_Krylov, r_THC)
        this_state.orthonormalize('right')
        #print(this_state.bond_dims)
        #orthogonalize "this state H \v_i" against the previous two”
        this_state = ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond_Krylov, trunc_tol)
        print(this_state.bond_dims)
        # store orthogonalized H \v_i in disk.
        filename = foldername + f"/Krylov_vec{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(this_state, file)
            
            
def ortho_to_previous_two(orth_state1, orth_state2, this_state, max_bond, trunc_tol_ortho):
    
    #orthoglnolize "this state" to previous two states, in MPS form.
    
    temp_state = copy.deepcopy(this_state)
    
    temp = copy.deepcopy(orth_state1)
    temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
    this_state = add_mps_and_compress(copy.deepcopy(this_state), temp, trunc_tol_ortho, max_bond)
    
    temp = copy.deepcopy(orth_state2)
    temp.A[0] = -vdot(temp_state, temp)* temp.A[0]
    this_state =  add_mps_and_compress(copy.deepcopy(this_state), temp, trunc_tol_ortho, max_bond)
    
    this_state.orthonormalize('right')
    
    return(this_state)


def get_W(N_use, foldername):
    #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
    #get W in the paper
    W = np.zeros([N_use, N_use])

    for i in range (N_use):
        for j in range (N_use):
            filename = foldername + f"/Krylov_vec{i}.pkl"
            with open(filename, 'rb') as file:
                temp1 = pickle.load(file)
                
            filename = foldername + f"/Krylov_vec{j}.pkl"
            with open(filename, 'rb') as file:
                temp2 = pickle.load(file)
            W[i,j] = np.vdot(temp1.as_vector(), temp2.as_vector())
    
    return(W)

def get_S(W):
    #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
    #get S using W in the paper
    x = W.shape[0]
    S = []
    S.append(np.zeros(x))
    S[0][0] = 1

    S_tilde = []
    S_tilde.append(np.zeros(x))
    S_tilde[0][0] = 1

    k = np.zeros([x,x])
    k[0, 0] = 1
    k[0,:] = W[:,0]

    Normalization = np.zeros(x)
    Normalization[0] = 1

    for n in range (1, x):
        
        S_tilde.append(np.zeros(x))
        S.append(np.zeros(x))
        
        S[n][n] = 1
        for i in range (n):
            for k in range (i+1):
                for k_prime in range (i+1):
                    S[n][k_prime] += -W[n,k] * S[i][k_prime]* S[i][k]     
                
        Normalization[n] = np.sqrt(sum(S[n][p] * S[n][q] * W[p, q] for p in range(n + 1) for q in range(n + 1)))

        S[n] = S[n]/Normalization[n]

    return (S)

def generate_re_ortho_space(N_use, W, foldername):
    #use stratege proposd in <Lanczos algorithm with Matrix Product States for dynamical correlation functions> to improve orthogonality
    #generate a list of post-orthogonalized Krylov vectors (in np.array)
    #Note: the vectors in this list are all np.array, we will have another MPS version.
    S = get_S(W)
    vector_list = []
    
    filename = foldername + f"/Krylov_vec{0}.pkl"
    with open(filename, 'rb') as file:
        shape_test = pickle.load(file)
    L = shape_test.nsites
    
    for i in range (N_use):
        temp2 = np.zeros([2**L], dtype = 'complex128')
        for j in range (i+1):
            filename = foldername + f"/Krylov_vec{j}.pkl"
            with open(filename, 'rb') as file:
                temp1 = pickle.load(file)
            temp2 += S[i][j]* temp1.as_vector()
        
        temp2 /= np.linalg.norm(temp2)
        vector_list.append(temp2)
    return(vector_list)

def generate_reduced_H(vector_list, H):
    H_reduced = np.zeros([len(vector_list), len(vector_list)])
    for i in range (len(vector_list)):
        for j in range (len(vector_list)):
            H_reduced[i, j] = np.vdot(vector_list[i], H@vector_list[j])
    return(H_reduced)

def popcount(x):
    return bin(x).count('1')

def generate_Hamiltonian_with_occupation_number(H, n):
    """
    Efficiently extract a Hamiltonian with a fixed occupation number from a given Hamiltonian H.
    """
    rows, cols = H.nonzero()
    size = H.shape
    H_occu = sparse.csr_matrix((size[0], size[0]), dtype=float)
    
    valid_indices = [(i, j) for i, j in zip(rows, cols) if popcount(i) == n and popcount(j) == n]
    
    if valid_indices:
        valid_rows, valid_cols = zip(*valid_indices)
        H_occu[valid_rows, valid_cols] = H[valid_rows, valid_cols]

    return H_occu
