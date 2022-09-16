###################################################################################################################
###
###  Generalized Gell-Mann Matrices
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to construct all GGMM matrices for some dimension D
###
####################################################################################################################

import numpy as np
import math
import itertools

# =================================== PARAMETRIZING UNITARY =======================================

def symmetrical_gmm(j,k,d):
    '''Constructs symmetrical ggm matrices.
    
        Parameters:
        j,k: Indices to be iterated over.
        d: Dimension.
        
        Returns: One symmetrical ggmm.'''

    A = np.zeros((d,d),int)
    A[k][j] = 1
    A[j][k] = 1 
    return A


def antisymmetrical_gmm(j,k,d):
    '''Constructs antisymmetrical ggm matrices.
    
        Parameters:
        j,k: Indices to be iterated over.
        d: Dimension.
        
        Returns: One antisymmetrical ggmm.'''

    A = np.zeros((d,d),complex)
    A[j][k] = -1j
    A[k][j] = 1j 
    return A


def construct_ggmm(lambdas,d):
    '''Constructs full set of symmetrical and  antisymmetrical ggmms.
    
        Parameters:
        lambdas: List of ggmms to append to.
        d: Dimension.
        
        Returns: Antisymmetrical and symmetrical ggmms.'''

    for k in range(1,d):
        for j in range(k):
            lambdas.append(symmetrical_gmm(j,k,d))
            lambdas.append(antisymmetrical_gmm(j,k,d))
    return lambdas


def diagonal_helper(l,d):
    '''Helper function to construct diagonal ggmms.'''

    A = np.zeros((d,d),int)
    for j in range(l+1):
        A[j][j]=1
    return A


def diagonal_ggmm(lambdas,d):
    '''Function to construct diagonal ggmms.
    
        Parameters:
        lambdas: List of ggmms to append to.
        d: Dimension.
        
        Returns: List of ggmms with diagonal ggmms appended.'''

    E_1 = np.zeros((d,d),int)
    for l in range(0, d-1):
        coeff = math.sqrt(2/((l+1)*(l+2)))
        E_1[l+1][l+1] = 1
        lambdas.append(coeff*((-(l+1)*E_1)+diagonal_helper(l,d)))
        E_1 = np.zeros((d,d),int)
    
    return lambdas


def construct_ggmm_sub(d):
    '''Constructs ggmms for a subsystem of dimension d.
    
        Parameters:
        d: Dimension.
        
        Returns: List of ggmms of dimension d.'''

    lambdas = []
    construct_ggmm(lambdas,d)
    diagonal_ggmm(lambdas,d)
    return lambdas


def tensor_Ib(matrix, ident_b):
    '''Tensors identity matrix of dimension of subsystem b with matrix.'''
    return np.kron(matrix, ident_b)


def tensor_Ia(matrix,ident_a):
    '''Tensors matrix with identity matrix of dimension of subsystem a.'''
    return np.kron(ident_a,matrix)


def construct_all_ggmm(d_a, d_b):
    '''Constructs ggmms for joint system.
    
        Parameters:
        d_a: Dimension of subsystem a.
        d_b: Dimension of subsystem b.
        
        Returns: Complete list of ggmms of dimensions d_a*d_b.'''
        
    total = []
    a_ggmm = construct_ggmm_sub(d_a)
    b_ggmm = construct_ggmm_sub(d_b)
    ident_b = np.identity(d_b)
    ident_a = np.identity(d_a)
    total.extend([tensor_Ib(x,ident_b) for x in a_ggmm])
    total.extend([tensor_Ia(x,ident_a) for x in b_ggmm])
    for a in a_ggmm:
        for b in b_ggmm:
            total.append(np.kron(a,b))
    return total

def tensor_func(n, tensorder):
    ''' Function that computes the tensor product of matrices for n qubits
    
        Parameters:
        n: number of environmental qubits
        tensorder: list of matrices to tensor 
        
        Returns: GGMMs of the correct dimension and interaction'''

    term = tensorder[0]
    for i in range(0, n):
        term = np.kron(term, tensorder[i+1])

    return term

def construct_all_ggmm_nm(d, envqus):
    '''New method for constructing ggmms for joint system while also keeping track of the GGMMs that belong to a certain system.
    
        Parameters:
        d: Dimension of qubits
        envqus: Number of environmental qubits
        
        Returns: Complete list of ggmms of dimensions d**total qubits.'''
        
    totqus = envqus + 1

    sys_ggmm = construct_ggmm_sub(d)      # List of sigma_x, sigma_y, sigma_z
    ident = np.identity(d)                # 2x2 Identity Matrix
    alpha = [ident, sys_ggmm[0], sys_ggmm[1], sys_ggmm[2]]
    
    # First creating list of lists to be tensored 
    x = [0, 1, 2, 3]                                                 # Each slot can vary between 0..3
    numlist = [p for p in itertools.product(x, repeat = totqus)]     # Applying a permutation with repetition 
    numlist.pop(0)                                                   # Getting rid of the first element because it is not traceable
    
    # Tesnoring the lists to construct each GGMM
    allGGMMs = []
    tenslist = []
    check = []
    for i in range(len(numlist)):
        for j in range(envqus+1):
            tenslist.append(alpha[numlist[i][j]])                   # Adding in the corresponding alpha matrix to the tensor list
            check.append(numlist[i][j])
        allGGMMs.append(tensor_func(envqus, tenslist))          # Adding in corresponding GGMM into the total list
        tenslist = []
        check = []
    
    return numlist, allGGMMs



# ========================= SYS GGMMS ============================

def construct_sys_ggmm(lambdas,d):
    '''Constructs full set of symmetrical and  antisymmetrical ggmms for the system.
    
        Parameters:
        lambdas: List of ggmms to append to.
        d: Dimension.
        
        Returns: Antisymmetrical and symmetrical ggmms.'''

    for k in range(1,d):
        for j in range(k):
            lambdas.append(symmetrical_gmm(j,k,d))
            lambdas.append(antisymmetrical_gmm(j,k,d))

    return lambdas


def construct_sys_ggmm_sub(d):
    '''Constructs ggmms for a subsystem of dimension d.
    
        Parameters:
        d: Dimension.
        
        Returns: List of ggmms of dimension d.'''

    lambdas = []
    construct_sys_ggmm(lambdas, d)
    diagonal_ggmm(lambdas, d)

    return lambdas


def construct_allsys_ggmm(d_a, d_b):
    '''Constructs ggmms for system.
    
        Parameters:
        d_a: Dimension of subsystem a.
        d_b: Dimension of subsystem b.
        
        Returns: Complete list of ggmms of dimensions d_a*d_b.'''
        
    total = []
    a_ggmm = construct_sys_ggmm_sub(d_a)
    ident_b = np.identity(d_b)
    total.extend([tensor_Ib(x,ident_b) for x in a_ggmm])

    return total

# ========================= UNITARY TESTS ============================

def correct_len(d_sys, d_env):
    #Verifies list returned by construct_all_ggmm is length D**2-1.
    D = d_sys*d_env
    return ((D**2-1) == len(construct_all_ggmm(d_sys,d_env)))
#print(correct_len())