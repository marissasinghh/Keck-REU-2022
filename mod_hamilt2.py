###################################################################################################################
###
###  Constructing Hamiltonian 2
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to construct the 1-D Ising Hamiltonian found in the Cotler text
###  -- this Hamiltonian describes a system where one qubit is coupled with n other qubtis in a chain-like, circular manner
###  -- for example, if n=2, you can imagine a triangular chain, or if n=4, you have a square chain 
###  -- note: this system has self-Hamiltonians
###  -- takes in three variables:
###  ----> n: number of coupled qubits
###  ----> J: strength of coupling 
###  ----> h: strength of self-Hamiltonian -- predefined as 0 for now
###  -- outputs H2
###
####################################################################################################################


import numpy as np

# Creating the needed Matrices
sigma_z = np.array([[1, 0],
                    [0, -1]])
sigma_x = np.array([[0, 1],
                    [1, 0]])
Id = np.identity(2)

# Although this Hamiltonian inlcudes self-Hamiltonians, for simplicity, we will start with:
h = 0


def construct_H2(n, J):
    '''Parameters:
        n: Number of environmental Qubits.
        J: Coupling Coefficient 

        Returns: Ising Model Hamiltonian.'''
 
    H_tot = np.zeros((2**(n+1), 2**(n+1)))

    ### ADDING IN SELF-HAMILTONIANS ### 

    term = h*sigma_x
    for i in range (0, n+1):                        # i corresponds to term number
        for j in range (0, n):                      # j corresponds to slot in term 
            if (j+1) == i:                          # same for term # + 1 = slot #, but only for terms after term 1 (i.e. i>1)
                term = np.kron(term, sigma_x)
            else:                                   # all other terms are tensored with identity 
                term = np.kron(term, Id)
        H_tot += term                               # add term to Hamiltonian                       
        term = h*Id                                 # after first term, first tensor changes to identity 


    ### ADDING IN INTERACTION HAMILTONIANS ###
    
    # term = J*sigma_z
    # for i in range (1, n+1):                        # i corresponds to term number
    #     for j in range (1, n+1):                    # j corresponds to slot in term 
    #         if j == i:                              # each nth term corresponds to the nth slot being coupled to the nth+1 slot
    #             term = np.kron(term, sigma_z)       # when term # = slot #, sigma tensors with wht exists
    #         elif (j+1) == i and i > 1:              # same for term # + 1 = slot #, but only for terms after term 1 (i.e. i>1)
    #             term = np.kron(term, sigma_z)
    #         else:                                   # all other terms are tensored with identity 
    #             term = np.kron(term, Id)
    #     H_tot += term                               # add term to Hamiltonian                       
    #     term = J*Id                                 # after first term, first tensor changes to identity 

    
    # NEW METHOD FOR BOUNDARY CONDITION 

    # First making the list of matrices to be tensored
    kronlist = [sigma_z, sigma_z]
    for j in range (1, n):
        kronlist.append(Id)

    # [term, term] // n = 1 ---> run 1 times 
    # [term, term, id] // n = 2 ---> run 3 times
    # [term, term, id, id] // n = 3 ---> run 4 times 

    if n>1:
        for i in range(0, n+1):                            # Runs number of total qubits times...

            term = kronlist[0]
            for k in range(0, n):                          # Runs number of env qubits times
                term = np.kron(term, kronlist[k+1])
            
            H_tot += -J*term

            # Changing order of list
            kronlist.insert(0, (kronlist.pop(n)))
    else:
        term = kronlist[0]
        for k in range(0, n):                             # Runs number of env qubits times
            term = np.kron(term, kronlist[k+1])
        
        H_tot += -J*term

    print("H2: ", H_tot)
    return(H_tot)
    

# TESTING 
# construct_H2(1, 1)
# construct_H2(2, 1)
# construct_H2(3, 1)
