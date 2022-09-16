###################################################################################################################
###
###  Creating Initial States
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to create the initial states of the env and the sys  
###  -- takes in two variables:
###  ----> n: number of coupled qubits
###  -- outputs collective initial state 
###
####################################################################################################################


import numpy as np 
import mod_createperm as cp


## Original Code ##

# Create initial state - 4x4 matrix with only "1" in the first entry
# A = np.zeros((d_sys, d_sys), int) 
# A[0][1] = 1  
# A[1][1] = 1
# test2 = np.kron(A, A)  
# print("Initial State: ", test2)


## New initial state code ##

def init_states(n):
    '''Parameters:
        n: Number of environmental Qubits.

        Returns: Initial state to be evolved forward in time.'''

    # Intial state for env will be randomly set -- OLD CODE MORE PHYSICALLY MOTIVATED 
    # conditions: has to be positive Hermitian matrix with trace = 1
    # A = np.zeros(((2**n), (2**n)), complex)
    # for row in range(0, n+1):
    #     for col in range(0, n+1):
    #         A[row][col] = (np.random.random() + np.random.random() * 1j)
    # B = np.dot(A, np.conjugate(np.transpose(A)))
    # env_i = (1/(np.trace(B)))*B

    # Initial state for env will be a maximally mixed state of the environment 
    # conditions: has to be positive Hermitian matrix with trace = 1
    env_i = (1/(2**n))*np.identity(2**n)

    # Creating list of the composite initial states!
    initial_states = []
    for i in range(2):
        sys_i = np.zeros((2, 2), int)
        sys_i[i][i] = 1
        state = np.kron(sys_i, env_i)
        initial_states.append(state)

    return initial_states


# # TESTING 
# a = init_states(1)
# b = init_states(2)
# c = init_states(3)