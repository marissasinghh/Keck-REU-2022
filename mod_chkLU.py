###################################################################################################################
###
###  Check Local Unitary Orbit  
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to check if the Hamiltonia found through the gradient descent algorithm
###  is from the same tensor product structure as the Native Hamiltonian 
###  -- does this by computing all of the checks found in Sun et. al's paper on Local Unitary Equivalence 
###  -- Note: code only works for two-qubit systems
###
####################################################################################################################
 

# H is a traceless 2**n x 2**n Hermitian matrix
# --> thus, H has the same expansion as the paramaterized sum of the GGMMs 
# before they are exponentiated and turned into the special Unitary Matrices 

# We can find the coefficients of this expansion by computing
# < sigma_alpha tensor Id, A > = tr((sigma_alpha tensor Id)*A)

# Ex. h_10 = Tr(GGMM*H)

import numpy as np 
from ast import literal_eval
import mod_hamilt1 as H1
import mod_hamilt2 as H2
import mod_hamilt3 as H3
import mod_hamilt4 as H4
import mod_createperm as cp

# Pauli Sigma Matrices
sigma_x = np.array([[0, 1],
                    [1, 0]])
sigma_y = np.array([[0, 0-1j],
                    [0+1j, 0]])
sigma_z = np.array([[1, 0],
                    [0, -1]])
sigma = [sigma_x, sigma_y, sigma_z]

Id = np.identity(2)

def getvals(H):

    h_self1 = []
    h_self2 = []
    h_intL = []
    
    # Creating list for H self 1
    for i in range(0,3):
        h_self1.append(np.trace(1/4*np.kron(sigma[i], Id).dot(H)))

    # Creating list for H self 2
    for j in range(0,3):
        h_self2.append(np.trace(1/4*np.kron(Id, sigma[j]).dot(H)))

    # Creating list for H int
    for k in range(0,3):
        for l in range(0,3):
            h_intL.append(np.trace(1/4*np.kron(sigma[k], sigma[l]).dot(H)))

    # Turning lists into arrays and matrices
    m = 0
    h_int = np.zeros((3,3), dtype = complex)
    for row in range(0,3):
        for col in range(0,3):
            h_int[row][col] = h_intL[m]
            m += 1


    # Series 1 checks
    mu = [0, h_self1, h_int.dot(h_self2), h_int.dot(np.transpose(h_int)).dot(h_self1), h_int.dot(np.transpose(h_int)).dot(h_int).dot(h_self2), 
          h_int.dot(np.transpose(h_int)).dot(h_int).dot(np.transpose(h_int)).dot(h_self1), h_int.dot(np.transpose(h_int)).dot(h_int).dot(np.transpose(h_int)).dot(h_int).dot(h_self2)]

    # Series 2 checks
    nu = [0, h_self2, np.transpose(h_int).dot(h_self1), np.transpose(h_int).dot(h_int).dot(h_self2)]

    # Compute 9 values that must be invariant for both H's

    # First 6 values
    checks = []
    for m in range(1,4):
        checks.append(np.inner(mu[m], mu[m]))
        checks.append(np.inner(nu[m], nu[m]))
        checks.append(np.inner(mu[1], mu[m*2]))

    # Last 3 values 
    checks.append(np.trace(h_int.dot(np.transpose(h_int))))
    checks.append(np.trace(h_int.dot(np.transpose(h_int)).dot(h_int).dot(np.transpose(h_int))))
    checks.append(np.linalg.det(h_int))

    rounded_checks = []
    for n in range(len(checks)):
        rounded_checks.append(round(checks[n], 5))

    
    return rounded_checks

    ############################# Make sure GGMMs and H have trace of 1

def compvals(Hnat, H):

    # First permutate H... onlt writing code for 2 qubits
    perm = cp.createperm(2, 2) 
    Hperm = (perm.dot(H).dot(np.conjugate(np.transpose(perm))))

    checks_Hnat = getvals(Hnat)
    checks_H = getvals(H)
    checks_Hperm = getvals(Hperm)
    for i in range(len(checks_H)):
        print("Value ", i, " for H_native: ", checks_Hnat[i])
    
    for i in range(len(checks_H)):
        print(" for H_test, QU1: ", checks_H[i])

    for i in range(len(checks_H)):
        print(" for H_test, QU2: ", checks_Hperm[i])


    # Sorting both the lists
    checks_Hnat.sort()
    checks_H.sort()
    checks_Hperm.sort()
    
    # Check if lists are equal 
    if checks_Hnat == checks_H == checks_Hperm:
        print ("H is equal to H_native")
    else :
        print ("H is not equal to H_native")


## Testing ##
# H2_nat = H2.construct_H2(1, 1)
# H2_scram = [[ (-0.54+0.00j), (-0.73+0.41j), (0.00+0.00j), (0.00+0.00j) ],
#             [ (-0.73-0.41j), (0.54+0.00j), (0.00+0.00j), (0.00+0.00j) ],
#             [ (0.00+0.00j), (0.00+0.00j), (0.54+0.00j), (0.73-0.41j) ],
#             [ (0.00+0.00j), (0.00+0.00j), (0.73+0.41j), (-0.54+0.00j) ]]
# compvals(H2_nat, H2_scram)



## Code to read in file and data ##

trial_num = int(input('Which Trial? '))
H_num = int(input('Hamiltonian Number? '))
q_num = int(input('Number of Total Qubits? '))

# Native Hamiltonian
cc = 1
if H_num == 1:
    H_native = H1.construct_H1(q_num-1, cc) 
elif H_num == 2:
    H_native = H2.construct_H2(q_num-1, cc)
elif H_num == 3:
    H_native = H3.construct_H3(q_num-1, cc)
elif H_num == 4:
    H_native = H4.construct_H4(q_num-1, cc)

filename = open(str(trial_num) + '_systrialEIG_H' + str(H_num) + '_qu' + str(q_num) + '.txt', 'r') 

lines = filename.readlines()
num_els = (2**q_num)**2
#M_els = literal_eval(lines[5])  # If file comes in from gradient descent 
M_els = literal_eval(lines[10]) # If file comes in from sysEIG

# Forming H_scram from file
n = 0
H_scram = np.zeros((2**q_num, 2**q_num), dtype = complex)
for row in range(0, 2**q_num):
    for col in range(0, 2**q_num):
        H_scram[row][col] = M_els[n]
        n += 1

# Run through definitions 
compvals(H_native, H_scram)

