###################################################################################################################
###
###  Constructing Hamiltonian 4
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to construct the Heisenberg Hamiltonian 
###  -- takes in three variables:
###  ----> n: number of coupled qubits
###  ----> c: strength of coupling 
###  -- outputs H4
###
####################################################################################################################


import numpy as np

# Creating the needed Matrices
sigma_x = np.array([[0, 1],
                    [1, 0]])
sigma_y = np.array([[0, 0-1j],
                    [0+1j, 0]])
sigma_z = np.array([[1, 0],
                    [0, -1]])
Id = np.identity(2)


def construct_H4(n, J):
    '''Parameters:
        n: Number of Qubits.
        J: Coupling Coefficient -- not in use, just here to match inputs of other 2 hamiltonians.

        Returns: Heisenberg Hamiltonian.'''
    
    H_tot = np.zeros((2**(n+1), 2**(n+1)), dtype = complex)
    terms = [sigma_x, sigma_y, sigma_z]
    
    for i in range(0, 3):    # Does this for every subterm 
        
        # First making the list of matrices to be tensored
        kronlist = [terms[i], terms[i]]
        for j in range (1, n):
            kronlist.append(Id)
        
        # [term, term] // n = 1 ---> run 1 times 
        # [term, term, id] // n = 2 ---> run 3 times
        # [term, term, id, id] // n = 3 ---> run 4 times 
                                                             
        if n>1:
            for i in range(0, n+1):                            # Runs number of total qubits times...

                term = kronlist[0]
                for k in range(0, n):                           # Runs number of env qubits times
                    term = np.kron(term, kronlist[k+1])
                
                H_tot += -J*term

                # Changing order of list
                kronlist.insert(0, (kronlist.pop(n)))
        else:
            term = kronlist[0]
            for k in range(0, n):                           # Runs number of env qubits times
                term = np.kron(term, kronlist[k+1])
            
            H_tot += -J*term

    print('H4: ', H_tot)      
    return H_tot

# TESTING 
# construct_H4(1, 1)
# construct_H4(2, 1)
# construct_H4(3, 1)

# # What we should get...

# q = -1*np.kron(sigma_x, sigma_x) + -1*np.kron(sigma_y, sigma_y)+ -1*np.kron(sigma_z, sigma_z)

# print("expected 2qu", q)

# e = -1*np.kron(np.kron(sigma_x, sigma_x), Id) + -1*np.kron(np.kron(sigma_y, sigma_y), Id) + -1*np.kron(np.kron(sigma_z, sigma_z), Id)
# f = -1*np.kron(np.kron(Id, sigma_x), sigma_x) + -1*np.kron(np.kron(Id, sigma_y), sigma_y) + -1*np.kron(np.kron(Id, sigma_z), sigma_z)
# g = -1*np.kron(np.kron(sigma_x, Id), sigma_x) + -1*np.kron(np.kron(sigma_y, Id), sigma_y) + -1*np.kron(np.kron(sigma_z, Id), sigma_z)

# print("expected 3qu", e+f+g )

# a = -1*np.kron(np.kron(np.kron(sigma_x, sigma_x), Id), Id) + -1*np.kron(np.kron(np.kron(sigma_y, sigma_y), Id), Id) + -1*np.kron(np.kron(np.kron(sigma_z, sigma_z), Id), Id)
# b = -1*np.kron(np.kron(np.kron(Id, sigma_x), sigma_x), Id) + -1*np.kron(np.kron(np.kron(Id, sigma_y), sigma_y), Id) + -1*np.kron(np.kron(np.kron(Id, sigma_z), sigma_z), Id)
# c = -1*np.kron(np.kron(np.kron(Id, Id), sigma_x), sigma_x) + -1*np.kron(np.kron(np.kron(Id, Id), sigma_y), sigma_y) + -1*np.kron(np.kron(np.kron(Id, Id), sigma_z), sigma_z)
# d = -1*np.kron(np.kron(np.kron(sigma_x, Id), Id), sigma_x) + -1*np.kron(np.kron(np.kron(sigma_y, Id), Id), sigma_y) + -1*np.kron(np.kron(np.kron(sigma_z, Id), Id), sigma_z)

# print("expected 4 qu", a+b+c+d)