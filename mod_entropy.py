###################################################################################################################
###
###  Entropy 
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to calculate the Von Neumann entropy and the purity entropy of a given 
###  density matrix 
###  
####################################################################################################################


import numpy as np
from numpy.linalg import matrix_power
from scipy.linalg import logm
 

def calculate_von_neumann_entropy(rho):
    '''Finds the Von Neumann entropy given a density operator.
    
        Parameters: 
        rho: density operator.
        
        Returns: Von neumann entropy.'''

    return (-np.matrix.trace(rho*logm(rho)))


def purity_entropy(rho):
    '''Finds the purity entropy given a density operator.
    
        Parameters: 
        rho: density operator.
        
        Returns: Purity entropy.'''
        
    return 1-np.matrix.trace(matrix_power(rho, 2))


# =============================== ENTROPY TESTS ===============================

# imports for testing
import mod_hamilt_OLD as h_old
import mod_reduce as red

def test_e_1():
    # Verifying expected maximal and minimal entropy values
    H_indv, H_total,w = h_old.construct_h_total(0)
    test1 = [[1/5, 0, 0, 0,0], [0,1/5,0,0,0], [0,0,1/5,0,0], [0,0,0,1/5,0], [0,0,0,0,1/5]]
    test2 = [[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    print(purity_entropy(test1))
    print(np.log(5))
    print("Maximally entangled:", round(calculate_von_neumann_entropy(test1), 13) == round(np.log(5), 13))
    print("Minimally entangled:", 0 == calculate_von_neumann_entropy(test2))


def test_e_2():
    # Verifying expected entropy value.
    test3 = [[1/9, 0, 0, 0,0,0,0,0,0], [0, 1/9,0, 0,0,0,0,0,0], [0,0, 1/9,0,0,0,0,0,0], [0, 0, 0, 1/9,0,0,0,0,0], [0, 0, 0, 0,1/9,0,0,0,0], [0, 0, 0, 0,0,1/9,0,0,0], [0, 0, 0, 0,0,0,1/9,0,0], [0, 0, 0, 0,0,0,0,1/9,0], [0, 0, 0, 0,0,0,0,0,1/9]]
    print("Result is:", calculate_von_neumann_entropy(test3) == np.log(9))


def test_e_3():
    # Verifying expected entropy value.
    whole_matrix = [[0,0,0,0], [0,1,0,0], [0,0,0,0], [0,0,0,0]]
    reduced = red.construct_reduced_density(whole_matrix, 2, 2, 'B')
    # Greater than 0
    print("Whole Matrix entropy = 0:", 0 == purity_entropy(whole_matrix))
    print("Reduced Matrix entropy > 0:", purity_entropy(reduced) > 0 )