# %%
import sys
sys.path.append('../')



import numpy as np
from pytenet.hartree_fock_mps import generate_single_state
from pytenet.operation import add_mps
from pytenet.hamiltonian_thc import eval_func, generate_thc_mpos_by_layer_qn, get_t, get_h1_spin, get_g_spin
from pytenet.global_krylov_method import generate_krylov_space_in_disk, get_W, get_S, generate_re_ortho_space_with_coeff, coeff_canonical_orthogonalization, remain_only_tridiagonal_elements
from pytenet.global_krylov_method import solve_ritz, generate_reduced_H_non_ortho, remain_only_tridiagonal_elements, coeff_gram_schmidt
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

# %% [markdown]
# Load and initialize datas: 
# 
# no is number of spatial orbitals
# 
# L is number of spinor orbitals, L = 2*no
# 
# t_spin is one-body integral in Chemist's notation (considering spins)
# 
# g_spin is two-body integral in Chemist's notation (considering spins)
# 
# X_mo and Z_mo are THC tensors, X_mo_up/down are X_mo considering spins
# 
# r_THC is THC rank

# %%
#load integrals
#with h5py.File("data_water/eri_water.hdf5", "r") as f:
with h5py.File("/work_fast/ge49cag/code_Luo/data/CO/integral.hdf5", "r") as f:
    eri = f["eri"][()]
    hkin = f["hkin"][()]
    hnuc = f["hnuc"][()]

#print(np.linalg.norm(eri))
#print(eri.shape)

no = eri.shape[0]
MV = eri.reshape(no*no,no*no)

u = np.load("/work_fast/ge49cag/code_Luo/data/CO/x.npy")
#u = np.load("/work_fast/ge49cag/pytenet_yu/water/x.npy")
X_mo = u.transpose(1,0)
g_thc, Z_mo = eval_func(u,eri,hkin,hnuc,)
h1 = hnuc+hkin
nmo = X_mo.shape[1]
L = 2*X_mo.shape[1]
g_thc = g_thc.reshape(nmo, nmo, nmo, nmo)
r_thc = X_mo.shape[0]

# %% [markdown]
# These Hamiltonian are exact molecular Hamiltonian and molecular Hamiltonian reconstructed by THC tensors. The calculation cost time, so that we store them in disk and load them when needed. For water molecule H2O in STO-6G basis, the error is small for r_THC = 28.
# 
# Actually, considering there are always 10 electrons for a water molecule, we only retain the elements which operator quantum states with 10 electrons.

# %% [markdown]
# Generate THC-MPO by layers, using THC tensors. 
# t_spin is used to create MPO for kinetic term.
# It returns a list of H_mu_nu, each H_mu_nu is also a list, which contains four smaller MPOs with bond dims 2.
# The final element of this list is MPO for kinetic term.

# %%
#generate thc_mpo
t = get_t(h1, eri)
H_mu_nu_list_spin_layer = generate_thc_mpos_by_layer_qn(X_mo, Z_mo, L, t)

print(type(H_mu_nu_list_spin_layer))
print(type(H_mu_nu_list_spin_layer[0]))
print(type(H_mu_nu_list_spin_layer[0][0]))
print((H_mu_nu_list_spin_layer[0][0].bond_dims))

# %% [markdown]
# Attention!!! Now mpo_ref is generated bt thc tensors!

# %%
g_phy =  eri.transpose(0, 2, 1, 3)
# mpo_ref = ptn.hamiltonian.spin_molecular_hamiltonian_mpo(h1, g_phy)
# print(mpo_ref.bond_dims)

g_thc_phy =  g_thc.transpose(0, 2, 1, 3)
mpo_ref = ptn.hamiltonian.spin_molecular_hamiltonian_mpo(h1, g_thc_phy)
print(mpo_ref.bond_dims)

#these values generated from Pyscf
e_ground = -135.0003866542846
e_1st_ex = -134.7673453653453
e_2nd_ex = -134.6729318042247

print(norm(g_phy - g_thc_phy))

# %% [markdown]
# For ground state finding, we use Hatree fock state |11111111110000> as initial state.
# 
# For 1st excited state, please use single-excited Hatree-Fock state as initial state, or even superposition of several single-excited Hatree-Fock states as initial state.

# %%
initial = generate_single_state(10, [3, 3, 3, 3, 3, 3, 3, 0, 0, 0])

# %% [markdown]
# We generate a group of orthogonal Krylov vectors using THC-MPO, with bond dim 40 for Krylov vectors. The vectors are stored in the folder = 'foldername', thus you don't have to generate them again for next time use. 

# %%
N_Krylov_1 = 30
#psi_original_1 = copy.deepcopy(initial)
max_bond_Krylov_1 = 250
trunc_tol = 1e-12
foldername_1 = f"/work_fast/ge49cag/code_datas/CO_Krylov"
#generate_krylov_space_in_disk(N_Krylov_1, H_mu_nu_list_spin_layer, psi_original_1, max_bond_Krylov_1, trunc_tol, r_thc, foldername_1)

# %%

H_reduced_non_rotho_1 = generate_reduced_H_non_ortho(N_Krylov_1, foldername_1, mpo_ref)
coeff_1 = coeff_gram_schmidt(N_Krylov_1, foldername_1)
H_reduced_1 = np.einsum('ik, kl, jl -> ij', coeff_1.conj(), H_reduced_non_rotho_1, coeff_1)

# %%
e_ritz_1, v_ritz_1 = solve_ritz(foldername_1, H_reduced_1, N_Krylov_1, coeff_1, max_bond_Krylov_1, e_ground, mpo_ref)

# %%
N_Krylov_2 = 30
#psi_original_2 = copy.deepcopy(v_ritz_1)
max_bond_Krylov_2 = 250
trunc_tol = 0
foldername_2= f"/work_fast/ge49cag/code_datas/CO_Krylov_2"
#generate_krylov_space_in_disk(N_Krylov_2, H_mu_nu_list_spin_layer, psi_original_2, max_bond_Krylov_2, trunc_tol, r_thc, foldername_2)



# %%
H_reduced_non_rotho_2 = generate_reduced_H_non_ortho(N_Krylov_2, foldername_2, mpo_ref)
coeff_2 = coeff_gram_schmidt(N_Krylov_2, foldername_2)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_2 = np.einsum('ik, kl, jl -> ij', coeff_2.conj(), H_reduced_non_rotho_2, coeff_2)


# %%
e_ritz_2, v_ritz_2 = solve_ritz(foldername_2, H_reduced_2, N_Krylov_2, coeff_2, max_bond_Krylov_2, e_ground, mpo_ref)

# %% [markdown]
# restart:

# %%
N_Krylov_3 = 20
#psi_original_3 = copy.deepcopy(v_ritz_2)
max_bond_Krylov_3 = 250
trunc_tol = 0
foldername_3 = f"/work_fast/ge49cag/code_datas/CO_Krylov_3"
#generate_krylov_space_in_disk(N_Krylov_3, H_mu_nu_list_spin_layer, psi_original_3, max_bond_Krylov_3, trunc_tol, r_thc, foldername_3)



# %%

H_reduced_non_rotho_3 = generate_reduced_H_non_ortho(N_Krylov_3, foldername_3, mpo_ref)
coeff_3 = coeff_gram_schmidt(N_Krylov_3, foldername_3)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_3 = np.einsum('ik, kl, jl -> ij', coeff_3.conj(), H_reduced_non_rotho_3, coeff_3)


# %%
e_ritz_3, v_ritz_3 = solve_ritz(foldername_3, H_reduced_3, N_Krylov_3, coeff_3, max_bond_Krylov_3, e_ground, mpo_ref)

# %% [markdown]
# restart:

# %%
N_Krylov_4 = 40
psi_original_4 = copy.deepcopy(v_ritz_3)
max_bond_Krylov_4 = 250
trunc_tol = 0
foldername_4 = f"/work_fast/ge49cag/code_datas/CO_Krylov_4"
generate_krylov_space_in_disk(N_Krylov_4, H_mu_nu_list_spin_layer, psi_original_4, max_bond_Krylov_4, trunc_tol, r_thc, foldername_4)


# %%
H_reduced_non_rotho_4 = generate_reduced_H_non_ortho(N_Krylov_4, foldername_4, mpo_ref)
coeff_4 = coeff_gram_schmidt(N_Krylov_4, foldername_4)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_4 = np.einsum('ik, kl, jl -> ij', coeff_4.conj(), H_reduced_non_rotho_4, coeff_4)


# %%
e_ritz_4, v_ritz_4 = solve_ritz(foldername_4, H_reduced_4, N_Krylov_4, coeff_4, max_bond_Krylov_4, e_ground, mpo_ref)

# %%
e_4, v_4 = np.linalg.eigh(H_reduced_4)
print('ideal value:', e_4[0] - e_ground)

# 试试扩大第三第四空间？

# %%
N_Krylov_5 = 30
psi_original_5 = copy.deepcopy(v_ritz_4)
max_bond_Krylov_5 = 250
trunc_tol = 0
foldername_5 = f"/work_fast/ge49cag/code_datas/CO_Krylov_5"
generate_krylov_space_in_disk(N_Krylov_5, H_mu_nu_list_spin_layer, psi_original_5, max_bond_Krylov_5, trunc_tol, r_thc, foldername_5)


# %%
H_reduced_non_rotho_5 = generate_reduced_H_non_ortho(N_Krylov_5, foldername_5, mpo_ref)
coeff_5 = coeff_gram_schmidt(N_Krylov_5, foldername_5)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_5 = np.einsum('ik, kl, jl -> ij', coeff_5.conj(), H_reduced_non_rotho_5, coeff_5)


# %%
e_ritz_5, v_ritz_5 = solve_ritz(foldername_5, H_reduced_5, N_Krylov_5, coeff_5, max_bond_Krylov_5, e_ground, mpo_ref)

# %%
e_5, v_5 = np.linalg.norm(H_reduced_5)
print('ideal value:', e_5[0] - e_ground)

# %%



