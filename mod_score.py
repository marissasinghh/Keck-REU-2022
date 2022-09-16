###################################################################################################################
###
###  Scoring
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to score the different tensor factrizations of the Hilbert Space  
###
####################################################################################################################


import math
import numpy as np 
from scipy.linalg import expm
import matplotlib.pyplot as plt 
import mod_reduce as red
import mod_timeevol as te
import mod_entropy as ent
import mod_inttests as intt


def step_vector(location):
    '''Given an np.array it perturbs the array a step away from the origin.
    
        Parameters:
        location: Array to be moved.
        
        Returns: Array with 0.0001 added to every entry.'''

    return location+0.0001


def step_vector_var(variance, D):
    '''Parameters:
        variance: Variance of outputted array.
        
        Returns: Array of size D**2-1 with given variance.'''

    #Loc represents the mean and scale the standard deviation, i.e. the square root of variance
    arr = np.random.normal(loc=0, scale=math.sqrt(variance), size=(D**2-1,1))
    return arr


def construct_unitary(thetas, GGMMs):
    '''Constructs unitary matrix from thetas.
    
        Parameters:
        thetas: Theta coefficients.
        
        Returns: Unitary matrix.'''

    param_GGMM = []
    for i in range(len(thetas)):
        param_GGMM.append(thetas[i]*GGMMs[i])
    
    #Sum all scaled matrices then exponentiate.
    scrambler = np.array(expm(1j*sum(param_GGMM)), dtype=complex)
    return scrambler 


# Do characteristic time up here
def scorer(thetas, ham, initial_state, GGMMs, n):
    '''Parameters:
        thetas: np.array of coefficients to construct unitary matrix from Generalized Gell-Mann matrices.
        lam: Interaction value between 0 and 1.
        ham: Hamiltonian operator to evolve state forward.
        initial_state: State to evolve forward.
        
        Returns: Entropy at the characteristic time for a given unitary matrix.'''
    
    # Specfiying dimensions 
    d_sys = 2
    d_env = (2)**n
    
    #Multiply the theta coefficients with the lambda matrices.
    param_GGMM = []
    for i in range(len(thetas)):
        param_GGMM.append(thetas[i]*GGMMs[i])
    
    #Sum all scaled matrices then exponentiate to construct unitary scrambler.
    scrambler = np.array(expm(1j*sum(param_GGMM)), dtype=complex)
    
    #Scramble Hamiltonian matrix.
    char_time = intt.characteristic_time(ham)
    H_total = (scrambler.dot(ham)).dot(np.conjugate(np.transpose(scrambler)))

    #Evolve intial state forward to characteristic time
    evolved_state = te.time_evolution(initial_state, H_total, char_time)

    #Take reduced density and find entropy.
    #return ent.purity_entropy(red.construct_reduced_density(evolved_state, d_sys, d_env, 'A')).real
    return ent.purity_entropy(red.reduce_DM_A(evolved_state, d_sys, d_env)).real



def scorer_avg(thetas, ham, n, env_i, GGMMs):
    '''Averages the scores produced by the system in every possible basis state and environment in the ready state.
    
        Parameters:
        thetas: Theta coefficient list.
        lam: Lambda coupling coefficient.
        ham: Hamiltonian operator.
        
        Returns: Average score. '''
    
    # Specifying dimensions
    d_sys = 2

    total = 0 
    # Original initial state of the env
    # B = (1/d_env)*np.ones((d_env,d_env))
   
    for i in range(d_sys):
        A = np.zeros((d_sys,d_sys),int)
        A[i][i]=1
        initial_state = np.kron(A, env_i)
        total += scorer(thetas, ham, initial_state, GGMMs, n)
        #print(total)
    return total/d_sys


def many_scores(start, steps, coup, ham, n, env_i, GGMMs):
    '''Function to see how entropy increases as theta moves away from origin in theta space.
    
        Parameters:
        start: Starting theta array.
        Steps: Steps to take away from origin.
        coup: Coupling coefficient.
        Ham: Hamiltonian operator.
        
        Returns: Plot of score vs distance from origin in theta space.'''
        
    scores = np.zeros(steps, dtype=complex)
    distances = np.array(range(steps))
    for i in range(steps):
        scores[i] = (scorer_avg(start, ham, n, env_i, GGMMs))
        start = step_vector_var(i*0.0001, 2*2**n)
    plt.plot(distances, scores, label = coup)
    plt.xlabel('Steps from Origin of Theta Space')
    plt.ylabel('Score')


# ========================= SYS SCORING ============================

def step_vector_var(variance, l):
    '''Parameters:
        variance: Variance of outputted array.
        
        Returns: Array of size D**2-1 with given variance.'''

    #Loc represents the mean and scale the standard deviation, i.e. the square root of variance
    arr = np.random.normal(loc=0, scale=math.sqrt(variance), size=(l,1))
    return arr