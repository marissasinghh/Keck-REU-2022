###################################################################################################################
###
###  Time Evolution 
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to time evolve a reduced density matrix 
###  
####################################################################################################################


from scipy.linalg import expm
from numpy.linalg import norm


# =============================== CHARACTERISTIC TIME ===============================

def characteristic_time(Hamiltonian):
    '''Calculates the characteristic time for a given Hamiltonian.
    
        Parameters: 
        Hamiltonian: The hamiltonian matrix.
        
        Returns: The characteristic time.'''

    #Time inversely related to energy
    return 1/(norm(Hamiltonian, 2))
    

# =============================== TIME EVOLUTION ===============================

def time_evolution(initial_state, H, t):
    '''Time evolution on density matrix.
    
        Parameters:
        initial state: State to evolve.
        H: Hamiltonian to use in evolution.
        t: Time to evolve to.
        
        Returns: Evolved state.'''
        
    return (expm(-(1j)*H*t).dot(initial_state)).dot(expm((1j)*H*t))


# =============================== TESTS TIME EVOLUTION ===============================

# Imports for testing
import numpy as np
import mod_hamilt_OLD as h_old

# C-NOT Hamiltonian
ham =np.matrix([[0,0,0,0], [0,0,0,0], [0,0,0.5,-0.5], [0,0,-0.5,0.5]])
#Confirm expected time evolution result using c-not Hamiltonian
#print(expm(-(1j)*ham*math.pi).dot([0,0,1,0]))

def test_separate():
    #Verifying that the evolution of both tensored together equals the evolution of the combined system
    A = [[0, 0, 0, 0,0], [0,1,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    B = [[1,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
    density_matrix = np.kron(A, B)
    H_smaller, H_total, w = h_old.construct_h_total(0)
    new_density_1 = time_evolution(density_matrix, H_total, 1)
    new_density_2 = np.kron(time_evolution(A, H_smaller, 1), time_evolution(B, H_smaller, 1))

    for i in range(len(new_density_1)):
        for j in range(len(new_density_1[0])):
            if round(new_density_1[i][j].real, 12) != round(new_density_2[i][j].real, 12):
                return "Test Failed"
    return "True"


# =============================== CHARACTERISTIC TIME ===============================

def characteristic_time(Hamiltonian):
    '''Calculates the characteristic time for a given Hamiltonian.
    
        Parameters: 
        Hamiltonian: The hamiltonian matrix.
        
        Returns: The characteristic time.'''

    # Time inversely related to energy
    return 1/(norm(Hamiltonian, 2))