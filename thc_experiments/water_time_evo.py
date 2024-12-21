# %%
import sys
sys.path.append('../')

import numpy as np
from pytenet.hartree_fock_mps import generate_single_state
from pytenet.hamiltonian_thc import eval_func, generate_thc_mpos_by_layer_qn, get_t, get_h1_spin, get_g_spin
from pytenet.global_krylov_method import generate_krylov_space_in_disk, get_W, get_S, remain_only_tridiagonal_elements
from pytenet.global_krylov_method import generate_Hamiltonian_with_occupation_number, generate_reduced_H_non_ortho, store_file, load_file
from pytenet.operation_thc import apply_thc_mpo_and_compress, add_mps_and_compress
from pytenet.operation import vdot, add_mps
import numpy as np
from scipy import sparse
import copy
import h5py
from numpy.linalg import norm
#np.set_printoptions(precision=4,suppress=True)
import scipy.io
import matplotlib.pyplot as plt
import pickle
import pytenet as ptn
import scipy.sparse.linalg as spla
from pytenet.krylov_time_evo import ED_time_evo, Krylov_evo_using_vecs_single_step, Krylov_time_evo_using_vecs, Krylov_evo_using_built_space, create_Krylov_space, gram_schmidt, Krylov_evo_using_built_mps_space

# %%
#load integrals
with h5py.File("/work_fast/ge49cag/pytenet_thc_spin_cons/data_water/eri_water.hdf5", "r") as f:
#with h5py.File("/work_fast/ge49cag/pytenet_yu/water/eri_water.hdf5", "r") as f:
    eri = f["eri"][()]
    hkin = f["hkin"][()]
    hnuc = f["hnuc"][()]

#print(np.linalg.norm(eri))
#print(eri.shape)

no = eri.shape[0]
MV = eri.reshape(no*no,no*no)

u = np.load("/work_fast/ge49cag/pytenet_thc_spin_cons/data_water/x.npy")
#u = np.load("/work_fast/ge49cag/pytenet_yu/water/x.npy")
X_mo = u.transpose(1,0)
g_thc, Z_mo = eval_func(u,eri,hkin,hnuc,)
h1 = hnuc+hkin
nmo = X_mo.shape[1]
L = X_mo.shape[1]
g_thc = g_thc.reshape(nmo, nmo, nmo, nmo)
r_thc = X_mo.shape[0]

# %%
# h1_spin = get_h1_spin(h1)
# g_spin = get_g_spin(eri)
g_phy =  eri.transpose(0, 2, 1, 3)
#mpo_ref = ptn.hamiltonian.molecular_hamiltonian_mpo(h1_spin, g_spin_phy)
mpo_ref = ptn.hamiltonian.spin_molecular_hamiltonian_mpo(h1, g_phy)
print(mpo_ref.bond_dims)

# %%
#ref Hamiltonian as matrix
H_ref = scipy.io.mmread('/work_fast/ge49cag/pytenet_thc_spin_cons/data_water/H_water_correct.mtx').tocsr()
H_thc = scipy.io.mmread('/work_fast/ge49cag/pytenet_thc_spin_cons/data_water/H_water_thc.mtx').tocsr()

#initial state
filename = f"/work_fast/ge49cag/code_datas" + f"/water_ground_ionization.pkl"
with open(filename, 'rb') as file:
    initial_state = pickle.load(file)

#thc mpo
t = get_t(h1, eri)
H_mu_nu_list_spin_layer = generate_thc_mpos_by_layer_qn(X_mo, Z_mo, L, t)
r_THC = int((len(H_mu_nu_list_spin_layer)-1)**(1/2) / 2)
    

# %%
T = 1
N_krylov = 4
# dt = 0.02 到 0.5
# 30：0.1到0.2要平滑
# 35：0.03到0.04
#dt_list = [0.02, 0.03, 0.0375, 0.04, 0.046875, 0.05, 0.08, 0.1, 0.125, 0.15, 0.1875, 0.2, 0.25, 0.3, 0.5]
#n_list = [round(T/dt) for dt in dt_list]
#n_list = [1,2,3,4,5,6,7,8,9,10,12,16,20,25,27,30,33,36,40,50,100]
n_list = [45,64,80]
#n_list = [200]
#n_list = [2, 4, 5]
dt_list = [T / n for n in n_list]
max_bond_list = [30, 35, 40]
foldername = f"/work_fast/ge49cag/code_datas/water_time_evo"
error_total_list = []

for max_bond in max_bond_list:
    
    print('max_bond', max_bond)
    
    error_list_bond = []
    
    for n in n_list:
        
        print(n)
        
        dt = T/n
        
        error_list_n = []
            
        for i in range(n):
            if i == 0:
                space = create_Krylov_space(N_krylov, H_mu_nu_list_spin_layer, copy.deepcopy(initial_state), 0, max_bond, r_THC)
                time_evolved_mps = Krylov_evo_using_built_mps_space(mpo_ref, space, max_bond, dt)
                #store_file(foldername, f"/N{N_krylov}B{max_bond}n{n}i{i}.pkl", time_evolved_mps)
                if i == n-1:
                    store_file(foldername, f"/N{N_krylov}B{max_bond}T{T}n{n}.pkl", time_evolved_mps)
                
            else:
                space = create_Krylov_space(N_krylov, H_mu_nu_list_spin_layer, copy.deepcopy(time_evolved_mps), 0, max_bond, r_THC)
                time_evolved_mps = Krylov_evo_using_built_mps_space(mpo_ref, space, max_bond, dt)
                if i == n-1:
                    store_file(foldername, f"/N{N_krylov}B{max_bond}T{T}n{n}.pkl", time_evolved_mps)

        psi_ed = ED_time_evo(H_ref, initial_state.as_vector(), T)
        psi_krylov_ref = Krylov_time_evo_using_vecs(H_ref, N_krylov, initial_state.as_vector(), n, T)
        
        trunc_error = norm(time_evolved_mps.as_vector() - psi_krylov_ref)
        krylov_error = norm(psi_krylov_ref - psi_ed)
        total_error = norm(time_evolved_mps.as_vector() - psi_ed)

        print(max_bond, n, dt)
        print('trunc error', trunc_error)
        print('krylov error', krylov_error )
        print('total error', total_error)
        
        # error_list_dt.append(trunc_error)
        # error_list_dt.append(krylov_error)
        # error_list_dt.append(total_error)
        # error_list_dt.append(dt)
        # error_list_dt.append(max_bond)
        
        error_list_n.extend([trunc_error, krylov_error, total_error, dt, max_bond])
    
        error_list_bond.append(error_list_n)
    
    error_total_list.append(error_list_bond)
    


# %%
# T = 0.2
# N_krylov = 4
# dt_list = [0.05, 0.1]
# max_bond_list = [30, 35]
# foldername = f"/work_fast/ge49cag/code_datas/water_time_evo"
# error_total_list = []

# for max_bond in max_bond_list:
    
#     error_list_bond = []
    
#     for dt in dt_list:
        
#         n = round(T/dt)
#         error_list_dt = []
        
#         time_evolved_mps = load_file(foldername, f"/Krylov_space{N_krylov}{max_bond}{dt}{n-1}.pkl")

#         psi_ed = ED_time_evo(H_ref, initial_state.as_vector(), T)
#         psi_krylov_ref = Krylov_time_evo_using_vecs(H_ref, N_krylov, initial_state.as_vector(), n, T)
        
#         trunc_error = norm(time_evolved_mps.as_vector() - psi_krylov_ref)
#         krylov_error = norm(psi_krylov_ref - psi_ed)
#         total_error = norm(time_evolved_mps.as_vector() - psi_ed)

#         print(max_bond, dt)
#         print('trunc error', trunc_error)
#         print('krylov error', krylov_error )
#         print('total error', total_error)
        
#         # error_list_dt.append(trunc_error)
#         # error_list_dt.append(krylov_error)
#         # error_list_dt.append(total_error)
#         # error_list_dt.append(dt)
#         # error_list_dt.append(max_bond)
        
#         error_list_dt.extend([trunc_error, krylov_error, total_error, dt, max_bond])
    
#         error_list_bond.append(error_list_dt)
    
#     error_total_list.append(error_list_bond)
    


# %%


# %%


# %%


# %%
# dt = 0.03

# Krylov_mps_list = copy.deepcopy(space_test)
# TN = np.zeros([len(Krylov_mps_list),len(Krylov_mps_list)])
# for i in range (TN.shape[0]):
#     for j in range (TN.shape[1]):
#         if abs(i - j) < 2:
#             TN[i, j] = ptn.operation.operator_inner_product(Krylov_mps_list[i], mpo_ref, Krylov_mps_list[j])
            
# c1 = np.zeros([len(Krylov_mps_list), 1])
# c1[0,0] = 1
# exp_TN = spla.expm(-1j*dt*TN)
# c_reduced = exp_TN@ c1

# psi_evloved = copy.deepcopy(Krylov_mps_list[0])
# psi_evloved.A[0] = c_reduced[0] *psi_evloved.A[0]

# for i in range (1, len(Krylov_mps_list), 1):
#     temp = copy.deepcopy(Krylov_mps_list[i])
#     temp.A[0] = c_reduced[i] *temp.A[0]
#     psi_evloved = add_mps(psi_evloved, temp)
    
# print(norm(psi_evloved.as_vector() - psi_krylov_ref))
# psi_evloved.orthonormalize('right')
# psi_evloved.orthonormalize('left')


# psi_evloved.compress_direct_svd_right_max_bond(0, max_bond)
# psi_evloved.orthonormalize('right')
# psi_evloved.orthonormalize('left')
    
# print(norm(psi_evloved.as_vector() - psi_krylov_ref))
# print(norm(psi_evloved.as_vector() - psi_ed))
# print(norm(psi_krylov_ref - psi_ed))

# %%
# /tmp/ipykernel_3032840/2463284093.py:8: ComplexWarning: Casting complex values to real discards the imaginary part
#   TN[i, j] = ptn.operation.operator_inner_product(Krylov_mps_list[i], mpo_ref, Krylov_mps_list[j])
# 3.5432687797238603e-06
# 9.462310845684574e-06
# 1.093342968130352e-05
# 5.4866900147348975e-06

# %%


# %%
# space_test_vec = []
# for i in range (len(space_test)):
#     temp = space_test[i].as_vector()
#     temp /= norm(temp)
#     space_test_vec.append(temp)

# space_test_vec = gram_schmidt(space_test_vec)

# space_test_vec = [mps_krylov.as_vector() for mps_krylov in space_test]
# space_test_vec = gram_schmidt(space_test_vec)

# %%
# #only use max_bond, don't set truncation tol!
# vec_test = Krylov_evo_using_built_space(H_ref, space_test_vec, 0.05)
# print(norm(vec_test - psi_krylov_ref))
# print(norm(vec_test - psi_ed))

# %%
# dt = 0.03

# D = 45:



# D = 40:
# 7.159356839258043e-13 (trunc error)
# 5.486690011281797e-06 (total error)

# D = 35
# 3.537471270532817e-06 (trunc error)
# 6.522437168647089e-06 (total error)

# D= 30
# 0.0004761841209985205 (trunc error)
# 0.00047618669594564364 (total error)

# %%


# %%



