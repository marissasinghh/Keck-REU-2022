###################################################################################################################
###
###  Constructing Hamiltonian 1
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to construct the Hamiltonian found in the Decoherence text
###  -- this Hamiltonian describes a system where one qubit is coupled with n other qubtis
###  -- note: this system has no self-Hamiltonians 
###  -- takes in two variables:
###  ----> n: number of environmental qubits
###  ----> g: strength of coupling 
###  -- outputs H1
###
####################################################################################################################


import numpy as np

# Creating the needed Matrices
sigma_z = np.array([[1, 0],
                    [0, -1]])
Id = np.identity(2)


def construct_H1(n, g):
    '''Parameters:
        n: Number of Qubits.
        g: Coupling coefficient. 

        Returns: Spin Bath Hamiltonian.'''

    H_tot = np.zeros((2**(n+1), 2**(n+1)))
    term = 1/2*sigma_z # Setting first slot of the every term (aka the system)

    for i in range (1, n+1):                        # i corresponds to term number
        for j in range (1, n+1):                    # j corresponds to slot in term
            if j < i or j > i:                      # term x Identity unless i = j
                term = np.kron(term, Id)
            else: # if i == j:                      # i = j, term x sigma_z
                term = np.kron(term, g*sigma_z)
        H_tot += term                               # add term to Hamiltonian                       
        term = 1/2*sigma_z                          # reset term 

    print("H1: ", H_tot)
    return(H_tot)


# TESTING 
# construct_H1(1, 1)
# construct_H1(2, 1)
# construct_H1(3, 1)