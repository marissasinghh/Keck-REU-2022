###################################################################################################################
###
###  Run Gradient Descnet Algorithm  
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  This version of the gradient descent algorithm turns the algorithm into a function that can be called in 
###  in runsysfile.py
###  -- intermediary step of code that allows things to run smoother  
###
####################################################################################################################


## Importing modules
import math
import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
import mod_score as sc
import mod_initstates as init
import mod_costfunc as cf
import mod_timeevol as te
import mod_createperm as cp
from scipy.linalg import expm 


def runsgd(trial_num, iterations, learning_rate, hamilt_num, numq, GGMMs, H_total, totGGMMs, ggmm_labels, adj_factor, newLR):
     
    # Opening files...
    output_file1 = open(str(trial_num) + '_systrialAVGN_H' + str(hamilt_num) + '_qu' + str(numq+1) + '.txt', 'w') 
    output_file2 = open('systrialAVGNscores_H' + str(hamilt_num) + '_qu' + str(numq+1) + '.txt', 'a') 
    # output_file4 = open('systrialAVGNSYS1scores_H' + str(hamilt_num) + '_qu' + str(numq+1) + '.txt', 'a')       
    output_file1.write("GRADIENT DESCENT TRIAL " + str(trial_num) + " FOR HAMILTONIAN " + str(hamilt_num) + '\n')
    output_file1.write("Running " + str(iterations) + " iterations with a learning rate of " + str(learning_rate) + " for " + str(numq+1) + ' total qubits \n')
    output_file1.write("*** Permutation matrices are being implemented! \n")
   
    # Scrambled Hamiltonian 
    scram_thetas = []
    for i in range(len(totGGMMs)):
        scram_thetas.append(np.random.uniform(-math.pi/4, math.pi/4))
    print('Scrambled theta key', scram_thetas)
    unitary = sc.construct_unitary(scram_thetas, totGGMMs)
    H_scram = (unitary.dot(H_total)).dot(np.conjugate(np.transpose(unitary)))
    char_time = te.characteristic_time(H_scram)

    # Writing scrambled info to file
    output_file1.write("\n Theta Key: \n")
    np.savetxt(output_file1, scram_thetas)

    # Implementing Permutation Matrices
    H_scrams = []
    H_scrams.append(H_scram)
    for i in range(1, numq+1):
        perm = cp.createperm(numq+1, i+1)
        H_scrams.append(perm.dot(H_scram).dot(np.conjugate(np.transpose(perm))))
        
    # Creating initial states -- list of [1 0][0 0] and [0 0][0 1] tensored states 
    # Only needed for original gradient descent method
    initial_states = init.init_states(numq)

    # New set of random thetas for starting iteration 
    theta0 = []
    for j in range(len(totGGMMs)):
        theta0.append(np.random.uniform(-math.pi/4, math.pi/4))
    print("Starting theta: ", theta0)

    # Writing starting info to file 
    output_file1.write("\n Theta Starting: \n")
    np.savetxt(output_file1, theta0)
    
    # Setting up initial states for the Matrix Method
    # System's initial state - the Pauli Sigma Matrices
    sigma_x = np.array([[0, 1],
                  [1, 0]])
    sigma_y = np.array([[0, 0-1j],
                        [0+1j, 0]])
    sigma_z = np.array([[1, 0],
                        [0, -1]])
    sigmas = [sigma_x, sigma_y, sigma_z]
    
    # Environment's initial state
    env_state = (1/(2**numq))*np.identity(2**numq)


    # ############### Implementing Machine Learning Alg ############### 
   
    # Cost Function - Original Gradient Descent 
    def cost_func_og(arb_theta):                               # basically f(a,b,c,...) = score
            score = cf.func1(GGMMs, H_scrams[1], numq, arb_theta, initial_states, char_time)
            return score

    # Cost Function - Matrix Method
    def cost_func(arb_theta):
        score = cf.func2(totGGMMs, H_scrams, arb_theta, sigmas, env_state, numq, char_time)
        return score
    
    ### QUICK CHECK OF SCORE AT THETA START
    startscore = cost_func(theta0)
    print("Check at starting theta :", startscore)
    output_file1.write("\n Theta Start Score: " + str(startscore) + "\n")
    
    # Graphing a slice of the cost function -- 3D graph 
    # Note -- graphs only work for original gradient descent method since 
    # matrix method outputs a score for an entire TPS
    # x = np.linspace(-math.pi, math.pi, 20)
    # y = np.linspace(-math.pi, math.pi, 20)

    # coords = []
    # vals = []
    # for a in range(len(x)):
    #     for b in range(len(y)):
    #         coords.append([x[a], y[b], 0])
    #         val = cost_func([x[a], y[b], 0])
    #         vals.append(val)
                    
    # # Reassigning xyz-vals according to coords
    # X = []
    # Y = []
    # Z = []
    # for i in range(len(coords)):
    #     X.append(coords[i][0])
    #     Y.append(coords[i][1])
    #     Z.append(vals[i])

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(X, Y, Z, cmap ='viridis', edgecolor ='green')
    # ax.set_xlabel('theta1')
    # ax.set_ylabel('theta2')
    # ax.set_zlabel('score')
    # ax.set_title('Hamiltonian %d' %(hamilt_num) + ' for %d' %(numq) + ' Env Qubits')
    # plt.show()
    


    # # ### GRAPH BLOCH SPHERE ###

    # # Cost Function - Matrix Method
    # def cost_func(arb_theta):
    #     score = cf.func2(totGGMMs, H_scrams, arb_theta, sigmas, env_state, numq, char_time)
    #     return score

    # # Graphing a slice of the cost function -- 3D graph 
    # x = np.linspace(-math.pi, math.pi, 20)
    # y = np.linspace(-math.pi, math.pi, 20)

    # coords = []
    # vals = []
    # for a in range(len(x)):
    #     for b in range(len(y)):
    #         coords.append([x[a], y[b], 0])
    #         val = cost_func([x[a], y[b], 0])
    #         vals.append(val)
                    
    # # Reassigning xyz-vals according to coords
    # X = []
    # Y = []
    # Z = []
    # for i in range(len(coords)):
    #     X.append(coords[i][0])
    #     Y.append(coords[i][1])
    #     Z.append(vals[i])

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_trisurf(X, Y, Z, cmap ='viridis', edgecolor ='green')
    # ax.set_xlabel('theta1')
    # ax.set_ylabel('theta2')
    # ax.set_zlabel('score')
    # ax.set_title('Hamiltonian %d' %(hamilt_num) + ' for %d' %(numq) + ' Env Qubits')
    # plt.show()
    
    # # Computing data 
    # x = np.linspace(-math.pi/4, math.pi/4, 20)
    # y = np.linspace(-math.pi/4, math.pi/4, 20)
    # z = np.linspace(-math.pi/4, math.pi/4, 20)

    # coords = []
    # vals = []
    # for a in range(len(x)):
    #     for b in range(len(y)):
    #         for c in range(len(z)):
          
    #             # Run through cost_func 
    #             arb_theta = [x[a], y[b], z[c]]
    #             scores, n_vec, val = cf.func2IO(totGGMMs, H_scrams, arb_theta, sigmas, env_state, numq, char_time)

    #             # QU1 as sys
    #             coords.append([n_vec[0][0], n_vec[0][1], n_vec[0][2]]) 
    #             vals.append(scores[0]) 
    #             print([n_vec[0][0], n_vec[0][1], n_vec[0][2]]) 

    #             # QU2 as sys
    #             # coords.append([n_vec[1][0], n_vec[1][1], n_vec[1][2]]) 
    #             # vals.append(vals[1]) 
                    
    # # Reassigning xyz-vals according to coords
    # X = []
    # Y = []
    # Z = []
    # V = []
    # for i in range(len(coords)):
    #     X.append(coords[i][0])
    #     Y.append(coords[i][1])
    #     Z.append(coords[i][2])
    #     V.append(vals[i])

    # # Create the figure, add a 3d axis, set the viewing angle
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(45,60)

    # img = ax.scatter(X, Y, Z, c=V, cmap=plt.hot())
    # fig.colorbar(img)

    # # Adding title and labels
    # ax.set_title("Heatmap of Bloch Sphere for H2 - QU1 as System")
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')

    # plt.show()

    # Opening file that documents cost history 
    output_file3 = open(str(trial_num) + '_systrialAVGN_CH_H' + str(hamilt_num) + '_qu' + str(numq+1) + '.txt', 'w')  

    # Starting Gradient Descent
    grad = nd.Gradient(cost_func)
    print("LR and its: ", learning_rate, iterations)
    theta_qc, cost_history = cf.gradient_descent(grad, theta0, learning_rate, iterations, cost_func, adj_factor, newLR)

    # Plotting the cost history against iterations to verify that we found the minimum
    fig,ax = plt.subplots()
    ax.set_ylabel('Score')
    ax.set_xlabel('Iterations')
    ax.set_title('Iterations vs Score')
    _ = ax.plot(range(iterations),cost_history,'b.')
    #plt.show()
    plt.close()
    
    # Writing cost history info to file
    for l in range(len(cost_history)):
        output_file3.write(str([l, cost_history[l]]) + '\n')
    output_file3.close()

    ### QUICK CHECK THAT AT THETA GUESS THERE IS A MINIMUM
    allscores, alln_vecs, endscore = cf.func2IO(totGGMMs, H_scrams, theta_qc, sigmas, env_state, numq, char_time)
    print("Check at theta guess: ", endscore)

    # Writing theta guess info to file
    output_file1.write("\n Theta Guess: \n")
    np.savetxt(output_file1, theta_qc)
    output_file1.write("\n ## Theta Final Guess Score: " + str(endscore) + " ## \n")
    output_file2.write(str([trial_num, endscore]) + '\n')

    output_file1.write("\n ## PERMUTATION INFO ## \n")
    # Writing individual N_vector and scoring info to file
    for i in range(len(allscores)):
        output_file1.write("\n N Vector for Qubit " + str(i+1) + ": \n")
        output_file1.write(str([alln_vecs[i][0], alln_vecs[i][1], alln_vecs[i][2]]) + "\n")
        output_file1.write("\n Corresponding Score for Qubit " + str(i+1) + ": \n")
        output_file1.write(str(allscores[i]) + '\n')

    # Closing files
    output_file1.close()
    output_file2.close()
    
    # Constructing the output H to be checked in the sys_eig method
    param_GGMM = []
    for i in range(0, len(theta_qc)):  # For sys gradient descent 
        param_GGMM.append(theta_qc[i]*totGGMMs[i])
    unitary_trial = np.array(expm(1j*sum(param_GGMM)), dtype=complex)
    H_new = (unitary_trial.dot(H_scram)).dot(np.conjugate(np.transpose(unitary_trial)))

    # Returning new H
    return H_new