import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from pytenet.hartree_fock_mps import generate_single_state

L = 6
hatree_state = generate_single_state(6, [0, 0 ,1 ,0, 0, 0])
for i in range (6):
    print(hatree_state.A[i])
    print('---')


