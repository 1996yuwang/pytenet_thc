# %%
import sys
sys.path.append('../')

import numpy as np
from pytenet.hartree_fock_mps import generate_single_state
from pytenet.hamiltonian_thc import eval_func, generate_thc_mpos_by_layer_qn, get_t, get_h1_spin, get_g_spin
from pytenet.global_krylov_method import generate_krylov_space_in_disk, get_W, get_S, remain_only_tridiagonal_elements
from pytenet.global_krylov_method import generate_Hamiltonian_with_occupation_number, generate_reduced_H_non_ortho, coeff_gram_schmidt, solve_ritz
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
# L is number of spatial orbitals, L = no
# 
# We DON'T need t_spin, g_spin now, only 'original' datas are enough 
# 
# r_THC is THC rank

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

# %% [markdown]
# These Hamiltonian are exact molecular Hamiltonian and molecular Hamiltonian reconstructed by THC tensors. The calculation cost time, so that we store them in disk and load them when needed. For water molecule H2O in STO-6G basis, the error is small for r_THC = 28.
# 
# Actually, considering there are always 10 electrons for a water molecule, we only retain the elements which operator quantum states with 10 electrons.

# %%
#load Hamiltonian generated by above coefficients
# H_correct = scipy.io.mmread('data_water/H_water_correct.mtx').tocsr()
# H_correct_10e = generate_Hamiltonian_with_occupation_number(H_correct.real, 10)
# e, v = sparse.linalg.eigsh(H_correct_10e, which = 'SA', k = 10)
# e_ground = e[0]
# e_1st_ex = e[1]

e_ground = -84.92262983120877
e_1st_ex = -84.52780562388604
e_2nd_ex = -84.46820215626565
e_3rd_ex = -84.42486181064207
e_4th_ex = -84.42379006099287

# %% [markdown]
# To construct 'original mpo' of molecular Hamiltonian, currently we can use ptn.hamiltonian.spin_molecular_hamiltonian_mpo(h1, g_phy), which consider the spins automatically, we don't need generating spin-version datas anymore. The result of ptn.hamiltonian.spin_molecular_hamiltonian_mpo(h1, g_phy) is mpo whose physical dimension is 4.

# %%
# h1_spin = get_h1_spin(h1)
# g_spin = get_g_spin(eri)
g_phy =  eri.transpose(0, 2, 1, 3)
#mpo_ref = ptn.hamiltonian.molecular_hamiltonian_mpo(h1_spin, g_spin_phy)
mpo_ref = ptn.hamiltonian.spin_molecular_hamiltonian_mpo(h1, g_phy)
print(mpo_ref.bond_dims)

# %% [markdown]
# Generate Hartree-Fock state. See function generate_single_state for further details.

# %%
HFS = generate_single_state(7, [3, 3, 3, 3, 3, 0, 0])

# %% [markdown]
# Generate THC-MPO by layers, using THC tensors. 
# It returns a list of H_mu_nu, each H_mu_nu is also a list, which contains four smaller MPOs with bond dims 2.
# The final element of this list is MPO for kinetic term.
# The physical bond_dim is 4.

# %%
#generate thc_mpo
t = get_t(h1, eri)
H_mu_nu_list_spin_layer = generate_thc_mpos_by_layer_qn(X_mo, Z_mo, L, t)

print(type(H_mu_nu_list_spin_layer))
print(type(H_mu_nu_list_spin_layer[0]))
print(type(H_mu_nu_list_spin_layer[0][0]))
print((H_mu_nu_list_spin_layer[0][0].bond_dims))

# %% [markdown]
# For ground state finding, we use Hatree fock state |11111111110000> as initial state.

# %%
#excited states for different molecules could have different total spins, please try different spin sections.
#for water in sto-6g: 1st excited state has spin 1
#HFS = generate_single_state(7, [3, 3, 3, 3, 3, 0, 0])
#HFS.orthonormalize('left')
#HFS.orthonormalize('right')

HFS = generate_single_state(7, [3, 3, 3, 3, 3, 0, 0])
HFS2 = generate_single_state(7, [3, 3, 3, 3, 1, 2, 0])
#HFS3 = generate_single_state(7, [3, 3, 3, 3, 1, 0, 2])
#HFS4 = generate_single_state(7, [3, 3, 3, 3, 0, 1, 2])

#initial = add_mps(HFS4, add_mps(HFS3, add_mps(HFS, HFS2)))
#initial =  add_mps(HFS3, add_mps(HFS, HFS2))
#initial =  add_mps(HFS4, add_mps(HFS, HFS2))
initial =  ptn.operation.add_mps(HFS, HFS2)
initial.orthonormalize('right')

# %%
#ptn.operation.operator_average(HFS2, mpo_ref) - e_1st_ex


# %% [markdown]
# We generate a group of orthogonal Krylov vectors using THC-MPO, with bond dim 40 for Krylov vectors. The vectors are stored in the folder = 'foldername', thus you don't have to generate them again for next time use. 

# %%
N_Krylov_1 = 40
psi_original = copy.deepcopy(initial)
max_bond_Krylov_1 = 30
#max_bond_Krylov = 75
trunc_tol = 0
foldername_1 = f"/work_fast/ge49cag/pytenet_thc_spin_cons/water_Krylov"
#Krylov vectors are included in data, you dont have to run generate it. ofc, you can -regenerate it to verify the algorithm using the following code:
#generate_krylov_space_in_disk(N_Krylov_1, H_mu_nu_list_spin_layer, psi_original, max_bond_Krylov_1, trunc_tol, r_thc, foldername_1)

#it indicates that even though during the calculation the bond dims exceed 40, but we only need 37 for Krylov vectors.

# %% [markdown]
# Make use of method proposed in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.85.205119 to improve the orthogonality of Krylov vectors. 

# %%
H_reduced_non_rotho_1 = generate_reduced_H_non_ortho(N_Krylov_1, foldername_1, mpo_ref)
coeff_1 = coeff_gram_schmidt(N_Krylov_1, foldername_1)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_1 = np.einsum('ik, kl, jl -> ij', coeff_1.conj(), H_reduced_non_rotho_1, coeff_1)

# %%
e_ritz_1, v_ritz_1 = solve_ritz(foldername_1, H_reduced_1, N_Krylov_1, coeff_1, max_bond_Krylov_1, e_ground, mpo_ref)

e_1, _ = np.linalg.eigh(H_reduced_1)
print(e_1[0] - e_ground)
print(e_1[1] - e_1st_ex)
# print(e_1[2] - e_2nd_ex)

# %%
N_Krylov_2 = 20
psi_original = copy.deepcopy(v_ritz_1)
max_bond_Krylov_2 = 30
trunc_tol = 0
foldername_2 = f"/work_fast/ge49cag/pytenet_thc_spin_cons/water_Krylov2"
generate_krylov_space_in_disk(N_Krylov_2, H_mu_nu_list_spin_layer, psi_original, max_bond_Krylov_2, trunc_tol, r_thc, foldername_2)

#it indicates that even though during the calculation the bond dims exceed 40, but we only need 37 for Krylov vectors.

# %%
H_reduced_non_rotho_2 = generate_reduced_H_non_ortho(N_Krylov_2, foldername_2, mpo_ref)
coeff_2 = coeff_gram_schmidt(N_Krylov_2, foldername_2)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_2 = np.einsum('ik, kl, jl -> ij', coeff_2.conj(), H_reduced_non_rotho_2, coeff_2)

# %%
e_ritz_2, v_ritz_2 = solve_ritz(foldername_2, H_reduced_2, N_Krylov_2, coeff_2, max_bond_Krylov_2, e_ground, mpo_ref)

# # %%
N_Krylov_3 = 20
psi_original = copy.deepcopy(v_ritz_2)
max_bond_Krylov_3 = 30
#max_bond_Krylov = 75
trunc_tol = 0
foldername_3 = f"/work_fast/ge49cag/pytenet_thc_spin_cons/water_Krylov3"
#Krylov vectors are included in data, you dont have to run generate it. ofc, you can -regenerate it to verify the algorithm using the following code:
generate_krylov_space_in_disk(N_Krylov_3, H_mu_nu_list_spin_layer, psi_original, max_bond_Krylov_3, trunc_tol, r_thc, foldername_3)

# #it indicates that even though during the calculation the bond dims exceed 40, but we only need 37 for Krylov vectors.

# # %%
H_reduced_non_rotho_3 = generate_reduced_H_non_ortho(N_Krylov_3, foldername_3, mpo_ref)
coeff_3 = coeff_gram_schmidt(N_Krylov_3, foldername_3)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_3 = np.einsum('ik, kl, jl -> ij', coeff_3.conj(), H_reduced_non_rotho_3, coeff_3)

# %%
e_ritz_3, v_ritz_3 = solve_ritz(foldername_3, H_reduced_3, N_Krylov_3, coeff_3, max_bond_Krylov_3, e_ground, mpo_ref)


# # %%
# # (0.35244435398342944+0j)
# # (0.024848970748564625+0j)
# # (0.0019585403953357172+0j)
# # (7.230799008084432e-05+0j)
# # (8.515112028817384e-07+0j)
# # (1.4224383448890876e-08+0j)
# # (1.0265921446261927e-10+0j)
# # (4.263256414560601e-14+0j)

# %%
N_Krylov_4 = 20
psi_original = copy.deepcopy(v_ritz_3)
max_bond_Krylov_4 = 30
#max_bond_Krylov = 75
trunc_tol = 0
foldername_4 = f"/work_fast/ge49cag/pytenet_thc_spin_cons/water_Krylov4"
#Krylov vectors are included in data, you dont have to run generate it. ofc, you can -regenerate it to verify the algorithm using the following code:
generate_krylov_space_in_disk(N_Krylov_4, H_mu_nu_list_spin_layer, psi_original, max_bond_Krylov_4, trunc_tol, r_thc, foldername_4)

# #it indicates that even though during the calculation the bond dims exceed 40, but we only need 37 for Krylov vectors.

# # %%
H_reduced_non_rotho_4 = generate_reduced_H_non_ortho(N_Krylov_4, foldername_4, mpo_ref)
coeff_4 = coeff_gram_schmidt(N_Krylov_4, foldername_4)
#H_reduced: elements calculated by post-orthogonalized Krylov vectos
H_reduced_4 = np.einsum('ik, kl, jl -> ij', coeff_4.conj(), H_reduced_non_rotho_4, coeff_4)

# %%
e_ritz_4, v_ritz_4 = solve_ritz(foldername_4, H_reduced_4, N_Krylov_4, coeff_4, max_bond_Krylov_4, e_ground, mpo_ref)

# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


# # %%


