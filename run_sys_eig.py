###################################################################################################################
###
###  Matrix Method Algorithm 
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  This algorithm takes in a tensor product structure and outputs the minimum score using the hand-written 
###  derivations of Dr. Kevin Setter 
###  -- takes in seven variables:
###  ----> trial_num: number of trial
###  ----> hamilt_num: number of Hamiltonian
###  ----> numq: number of environmental qubits
###  ----> sysGGMMs: system GGMMs
###  ----> H_total: Hamiltonian corresponding to the hamilt_num 
###  ----> totGGMMs: all GGMMs
###  ----> H_scram: scrambled Hamiltonian passed from the larger gradient descent algorithm 
###  -- saves the minimum score and relevant info to file
###
####################################################################################################################


## Importing modules
import math
import numpy as np
from numpy import linalg as LA
import mod_score as sc
import mod_createperm as cp
import mod_timeevol as te
import mod_reduce as red


def runse(trial_num, hamilt_num, numq, sysGGMMs, H_total, totGGMMs, H_scram):
    
    # Opening files...
    output_file1 = open(str(trial_num) + '_systrialEIG_H' + str(hamilt_num) + '_qu' + str(numq+1) + '.txt', 'w') 
    output_file2 = open('systrialEIGscores_H' + str(hamilt_num) + '_qu' + str(numq+1) + '.txt', 'a')  
    # output_file3 = open('systrialEIGSYS1scores_H' + str(hamilt_num) + '_qu' + str(numq+1) + '.txt', 'a')      
    output_file1.write("MATRIX METHOD TRIAL " + str(trial_num) + " FOR HAMILTONIAN " + str(hamilt_num) + '\n')
    output_file1.write("Running for " + str(numq+1) + ' total qubits \n')
    output_file1.write("*** Permutation matrices are being implemented! \n")

    # # Scrambled Hamiltonian 
    # scram_thetas = []
    # for i in range(len(sysGGMMs)):  # Change only this to sysGGMMs for native TPS scrambling 
    #     scram_thetas.append(np.random.uniform(-math.pi/4, math.pi/4))
    # print('Scrambled theta key', scram_thetas)
    # unitary = sc.construct_unitary(scram_thetas, totGGMMs)
    # H_scram = (unitary.dot(H_total)).dot(np.conjugate(np.transpose(unitary)))
    
    # # Writing scrambled info to file
    # output_file1.write("\n Theta Key: \n")
    # np.savetxt(output_file1, scram_thetas)

    # First write passed H_scram to file 
    print(H_scram)
    H_els = []
    for row in range(0, 2**(numq+1)):
        for col in range(0, 2**(numq+1)):
            H_els.append(H_scram[row][col])
    print(H_els)
    output_file1. write("\n Scrambled Hamiltonian Passed From Gradient Descent: \n")
    output_file1.write(str(H_els) + '\n')

    # Compute characteristic time
    char_time = te.characteristic_time(H_scram)

    # Implementing Permutation Matrices
    H_scrams = []
    H_scrams.append(H_scram)
    for i in range(1, numq+1):
        perm = cp.createperm(numq+1, i+1)
        H_scrams.append(perm.dot(H_scram).dot(np.conjugate(np.transpose(perm))))
       

    # System's initial state - the Pauli Sigma Matrices
    sigma_x = np.array([[0, 1],
                        [1, 0]])
    sigma_y = np.array([[0, 0-1j],
                        [0+1j, 0]])
    sigma_z = np.array([[1, 0],
                        [0, -1]])
    sigma = [sigma_x, sigma_y, sigma_z]
    
    # Environment's initial state
    env = (1/(2**numq))*np.identity(2**numq)


    ############### Implementing Matrix Method ############### 

    totscores = []
    for j in range(0, numq+1):
        
        print("###################### CHECKS FOR SYS AS QUBIT ", j+1,": ######################")
        output_file1.write("\n\n ## INFO FOR SYS AS QUBIT " + str(j+1) + " ## \n")

        # Computing Q alphas...
        Q_alphas = []
        for k in range(0, 3):        # For alpha = 1, 2, 3
            state = np.kron(sigma[k], env)
            Q_alphas.append(te.time_evolution(state, H_scrams[j], char_time))

        # Computing Matrix elements
        M = np.zeros((3,3), dtype = complex)
        for row in range(0,3):
            for col in range(0,3):
                a = np.array(red.reduce_DM_A(Q_alphas[row], 2, 2**numq))
                b = np.array(red.reduce_DM_A(Q_alphas[col], 2, 2**numq))
                M[row][col] = np.trace(a.dot(b))
        
        # Computing max eigenvalue and corresponding eigenvector 
        lam, V = LA.eigh(M)
        print("Eigenvalues", lam)
        score = lam[-1]
        print("MAX: ", max(lam), "SCORE LAM: ", score)
        n_vector = V[-1]
        
        score = 1/2-1/4*(score)

        # Checking score and corresponding vector on the Bloch Sphere
        print("N Vector: ", n_vector)
        print("Score: ", score)
        totscores.append(score)

        # if j == 0:
        #     output_file3.write(str([trial_num, score]) + '\n')
        # output_file3.close()

        # Writing info to file
        output_file1.write("\n N vector: \n")
        output_file1.write(str([n_vector[0], n_vector[1], n_vector[2]]) + '\n')
        output_file1.write("\n Score: " + str(score) + "\n")
    
    # Writing final averaged info to file 
    thetafinal = sum(totscores)/len(totscores)
    print("Final Score: ", thetafinal)
    output_file1.write("\n\n ## Final Averaged Score: " + str(thetafinal) + " ## \n")
    output_file2.write(str([trial_num, thetafinal]) + '\n')

    # Closing files
    output_file1.close()
    output_file2.close()
