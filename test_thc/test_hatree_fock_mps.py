import os
import sys
sys.path.append(os.getcwd())

import unittest
import numpy as np
from pytenet.hartree_fock_mps import generate_single_state
from pytenet.fermi_ops import generate_fermi_operators

class TestHartreeFock(unittest.TestCase):

    def test_hartree_fock_state(self):
        L = 6
        hartree_state = generate_single_state(6, [1, 1, 1, 1, 0, 0])
        for i in range(6):
            print(hartree_state.A[i])
            print('---')
            
        c_list, a_list = generate_fermi_operators(6)

        hartree_state_anni = a_list[3] @ hartree_state.as_vector()

        hartree_state_anni_mps = generate_single_state(6, [1, 1, 1, 0, 0, 0])

        self.assertTrue(np.linalg.norm(hartree_state_anni - hartree_state_anni_mps.as_vector()) == 0)

if __name__ == '__main__':
    unittest.main()
