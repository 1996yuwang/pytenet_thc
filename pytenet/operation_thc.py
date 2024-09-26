import numpy as np
from .operation import apply_operator, add_mps_and_compress, apply_operator_and_compress
import copy


def get_h1_spin(h1):
    
    nmo = h1.shape[0]
    
    h1_upup = np.zeros([2*h1.shape[0], 2*h1.shape[1]], dtype = h1.dtype)
    h1_dd = np.zeros([2*h1.shape[0], 2*h1.shape[1]], dtype = h1.dtype)

    for i in range (nmo):
        for j in range (nmo):
            h1_upup[2*i, 2*j] = h1[i, j]

    for i in range (nmo):
        for j in range (nmo):
            h1_dd[2*i+1, 2*j+1] = h1[i, j]

    h1_spin = h1_upup + h1_dd
    
    return(h1_spin)

def get_g_spin(g_mo):
    
    nmo = g_mo.shape[0]
    
    g_spin = np.zeros([2*nmo, 2*nmo, 2*nmo, 2*nmo], dtype = g_mo.dtype)

    for p in range(2*nmo):
        for q in range(2*nmo):
            for r in range(2*nmo):
                for s in range(2*nmo):
                    if p%2 == q%2:
                        if r%2 == s%2:
                            g_spin[p,q,r,s] = g_mo[p//2, q//2, r//2, s//2]
                            
    return(g_spin)
    
    
def get_t_spin(h1, g_mo):

    nmo = g_mo.shape[0]
    
    g_trace_temp = np.zeros([nmo, nmo])
    for p in range (nmo):
        for s in range (nmo):
            for i in range (nmo):
                g_trace_temp[p, s] += g_mo[p, i, i, s] 
                
    t = h1 - 0.5 * g_trace_temp

    t_upup = np.zeros([2*t.shape[0], 2*t.shape[1]], dtype = t.dtype)
    t_dd = np.zeros([2*t.shape[0], 2*t.shape[1]], dtype = t.dtype)

    for i in range (nmo):
        for j in range (nmo):
            t_upup[2*i, 2*j] = t[i, j]

    for i in range (nmo):
        for j in range (nmo):
            t_dd[2*i+1, 2*j+1] = t[i, j]

    t_spin = t_upup + t_dd
    
    return(t_spin)

def get_X_up(X_mo):
    r_THC = X_mo.shape[0]
    nmo = X_mo.shape[1]
    X_mo_up = np.zeros([X_mo.shape[0], 2*X_mo.shape[1]], dtype = X_mo.dtype)
    for i in range (r_THC):
        for j in range (nmo):
            X_mo_up[i, 2*j] = X_mo[i, j]
            
    return X_mo_up


def get_X_down(X_mo):
    r_THC = X_mo.shape[0]
    nmo = X_mo.shape[1]
    X_mo_down = np.zeros([X_mo.shape[0], 2*X_mo.shape[1]], dtype = X_mo.dtype)
    for i in range (r_THC):
        for j in range (nmo):
            X_mo_down[i, 2*j + 1] = X_mo[i, j]
    
    return X_mo_down

def H_on_mps_compress_by_layer(H_mu_nu_by_layer, psi, tol, max_bond_layer):
    ''' 
    H_mu_nu_by_layer: the four MPOs of sub-Hamiltonian H_mu_nu.
    Apply the four elementary MPOs to psi and compress, sequentially.
    return: compressed H_mu_nu|\psi>
    '''
    temp = copy.deepcopy(psi)
    for layer in H_mu_nu_by_layer:
        temp = apply_operator(layer, temp)
        temp.compress_no_normalization_max_bond(tol, max_bond = max_bond_layer)
    return temp


def apply_thc_mpo_on_and_compress(sub_H_list_as_layer, psi, trunc_tol, max_bond_global, r_THC):
    #svd_small_count = 0
    psi_original = copy.deepcopy(psi)
    for nu in range (r_THC):
        for s1 in range (2):
            #print("nu, s1:", nu, s1)
            H_nu_layers = sub_H_list_as_layer[find_indices_spin(0, nu, s1 ,0 ,r_THC)]
            temp_layers_nu = [H_nu_layers[0], H_nu_layers[1]]
            try:
                H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, trunc_tol, max_bond_global)
            except Exception:
                svd_small_count += 1
                print(nu, s1)
                try:
                    H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, 10*trunc_tol, max_bond_global)
                except Exception:
                    try:
                        H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, 100*trunc_tol, max_bond_global)
                    except Exception:
                        try:
                            H_nu_psi =  H_on_mps_compress_by_layer(temp_layers_nu, psi_original, 1000*trunc_tol, max_bond_global)
                        except Exception:
                            print("still fail for 4th attempt")
                    

            for mu in range(r_THC):
                for s2 in range (2):
                    #print("mu, s2:", mu, s2)
                    H_mu_layers = sub_H_list_as_layer[find_indices_spin (mu, nu, s1 ,s2 ,r_THC)]
                    temp_layers_mu = [H_mu_layers[2], H_mu_layers[3]]
                    if mu == 0 and s1 == 0 and nu == 0 and s2 == 0:
                        try:
                            H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, trunc_tol, max_bond_global)
                
                        except Exception:
                            svd_small_count += 1
                            print(mu, nu, s1, s2)
                            try:
                                H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 10*trunc_tol, max_bond_global)
                            except Exception:
                                try:
                                    H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 100*trunc_tol, max_bond_global)
                                except Exception:
                                    try:
                                        H_on_psi = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 1000*trunc_tol, max_bond_global)
                                    except Exception:
                                        print("still fail for 4th attempt")
                    else: 
                        try:
                            temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, trunc_tol, max_bond_global)
                            H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, trunc_tol, max_bond_global) 
                        except Exception:
                            svd_small_count += 1
                            print(mu, nu, s1, s2)
                            try:
                                temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 10*trunc_tol, max_bond_global)
                                H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, 10*trunc_tol, max_bond_global) 
                            except Exception:
                                try:
                                    temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 100*trunc_tol, max_bond_global)
                                    H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, 100*trunc_tol, max_bond_global) 
                                except Exception:
                                    try:
                                        temp_mps = H_on_mps_compress_by_layer(temp_layers_mu, H_nu_psi, 1000*trunc_tol, max_bond_global)
                                        H_on_psi = add_mps_and_compress(H_on_psi, temp_mps, 1000*trunc_tol, max_bond_global) 
                                    except Exception:
                                        print("still fail for 4th attempt")
    
    Kinetic_on_psi = apply_operator_and_compress(sub_H_list_as_layer[-1][0], psi_original, trunc_tol, max_bond_global)                         
    H_on_psi = add_mps_and_compress(H_on_psi, Kinetic_on_psi, trunc_tol, max_bond_global) 
     
    #wrong count and display here! 
    #print('SVD error caught:', svd_small_count, 'for:', mu, nu, s1, s2, )
    return (H_on_psi)



def find_indices_spin (mu, nu, s1 ,s2 ,r_THC):
    return (mu* 4* (r_THC) + nu* 4 + 2*s1 + s2)