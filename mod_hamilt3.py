###################################################################################################################
###
###  Constructing Hamiltonian 3
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to construct the C-NOT Hamiltonian 
###  -- takes in three variables:
###  ----> n: number of environmental qubits
###  ----> c: strength of coupling 
###  -- outputs H3
###
####################################################################################################################


import numpy as np

# Creating the needed Matrices
state0 = np.array([[1, 0],
                   [0, 0]])
state1 = np.array([[0, 0],
                   [0, 1]])
sigma_x = np.array([[0, 1],
                    [1, 0]])
Id = np.identity(2)


def construct_H3(n, c):
    '''Parameters:
        n: Number of Qubits.
        c: Coupling Coefficient -- not in use, just here to match inputs of other 2 hamiltonians.

        Returns: C-NOT Hamiltonian.'''
    
    H_tot = np.zeros((2**(n+1), 2**(n+1)))
    term1 = Id
    term2 = sigma_x

    for i in range(1, n+1):                    # i corresponds to term number
        if i == n:                             # last term 
            term1 = np.kron(state0, term1)     # for term1 must be state0 x term1
            term2 = np.kron(state1, term2)     # for term2 must be state1 x term2
        else:                                  # for n qubits, take
            term1 = np.kron(Id, term1)         # Identity matrix x term1 where term starts off as Identity Matrix
            term2 = np.kron(Id, term2)         # Identity matrix x term2 where term starts off as sigma_x 
    H_tot += term1 + term2                     # final H is the sum of these two terms
    
    print("H3: \n", H_tot)
    return(H_tot)


CNOT12 = np.kron(state0, Id) + np.kron(state1, sigma_x)
CNOT13 = np.kron(Id, np.kron(Id, state0)) + np.kron(sigma_x, np.kron(Id, state1)) 
# print("cnot12", CNOT12)
# print("cnot13", CNOT13)

# # TESTING 
# construct_H3(1, 1)
# construct_H3(2, 1)
# construct_H3(3, 1)