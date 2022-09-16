###################################################################################################################
###
###  Cost function
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to create the function that will be put into the machine learning algorithm
###  -- note: all thetas must be arbitrary so that the gradient can be properly taken
###  ----> thus each arbitrary theta is a letter of the alphabet 
###  -- takes one variable:
###  ----> GGMMs: the Generalized Gell Mann Matrices of the system
###  -- outputs cost function 
###
####################################################################################################################

import numpy as np
from scipy.linalg import expm 
from numpy import linalg as LA
import mod_inttests as intt
import mod_initstates as init
import mod_timeevol as te
import mod_reduce as red
import mod_entropy as ent


# Creating the cost function --- USED IN ORIGINAL GRADIENT DESCENT 
def func1(GGMM, scramham, n, arb_thetas, init_states, time):
    '''Parameters:
        GGMM: List of Generalized Gell-Mann Matrices.
        scramham: Scrambled Hamiltonian.
        n: Number of environmental qubits.
        arb_thetas: List of theta values that parameterize the Unitary Matrix.
        init_states: State to evolve forward.
        time: Characteristic time

        Returns: Score.'''
    
    # Specfiying dimensions 
    dsys = 2
    denv = (2)**n
    D = ((dsys*denv)**2)-1

    # Multiplying arbitrary thetas to the GGMMs
    param_GGMM = []
    for i in range(0, len(arb_thetas)):  
        param_GGMM.append(arb_thetas[i]*GGMM[i])
    
    # Constructing the trial unitary matrix with the GGMMs
    unitary_trial = np.array(expm(1j*sum(param_GGMM)), dtype = complex) 
    
    scores = []
    # Code for testing all scramhams
    # for i in range(0, len(scramham)):
    #     H_total = (unitary_trial.dot(scramham[i])).dot(np.conjugate(np.transpose(unitary_trial)))
    #     for i in range(0, dsys):
    #         evolved_state = te.time_evolution(init_states[i], H_total, time)
    #         scores.append(ent.purity_entropy(red.reduce_DM_A(evolved_state, dsys, denv)).real)
    
    # Code for only testing one hamiltonian 
    H_total = (unitary_trial.dot(scramham)).dot(np.conjugate(np.transpose(unitary_trial)))
    for i in range(0, dsys):
        evolved_state = te.time_evolution(init_states[i], H_total, time)
        scores.append(ent.purity_entropy(red.reduce_DM_A(evolved_state, dsys, denv)).real)

    # Return average score of both starting states 
    return sum(scores)/len(scores)


# Creating the new cost function --- MATRIX METHOD
def func2(GGMM, scramhams, arb_thetas, sigma, env, n, time):
    '''Parameters:
        GGMM: List of Generalized Gell-Mann Matrices.
        scramham: Scrambled Hamiltonian.
        n: Number of environmental qubits.
        arb_thetas: List of theta values that parameterize the Unitary Matrix.
        init_states: State to evolve forward.
        time: Characteristic time.

        Returns: Score.'''
    
    # Specfiying dimensions 
    dsys = 2
    denv = (2)**n
    D = ((dsys*denv)**2)-1

    # Multiplying arbitrary thetas to the GGMMs
    param_GGMM = []
    for i in range(0, len(arb_thetas)):              # sys 2
        param_GGMM.append(arb_thetas[i]*GGMM[i])
    
    # Constructing the trial unitary matrix with the GGMMs
    unitary_trial = np.array(expm(1j*sum(param_GGMM)), dtype = complex) 

    totscores = []
    # For each permutated matrix...
    for j in range(0, n+1):

        H_total = (unitary_trial.dot(scramhams[j])).dot(np.conjugate(np.transpose(unitary_trial)))
                
        # Computing Q alphas...
        Q_alphas = []
        for k in range(0, 3):        # For alpha = 1, 2, 3
            state = np.kron(sigma[k], env)
            Q_alphas.append(te.time_evolution(state, H_total, time))

        # Computing Matrix elements
        M = np.zeros((3,3), dtype = complex)
        for row in range(0,3):
            for col in range(0,3):
                a = np.array(red.reduce_DM_A(Q_alphas[row], 2, 2**n))
                b = np.array(red.reduce_DM_A(Q_alphas[col], 2, 2**n))
                M[row][col] = np.trace(a.dot(b))
        
        # Computing max eigenvalue and corresponding eigenvector 
        lam, V = LA.eigh(M)
        maxlam = lam[-1]

        # Adding important info into lists
        totscores.append(1/2-1/4*(maxlam))
    
    # Computing final score 
    finalscore = sum(totscores)/len(totscores)

    return finalscore

def func2IO(GGMM, scramhams, arb_thetas, sigma, env, n, time):
    ''' Same as func2, but saves n_vecs and other important info 
    
    Parameters:
        GGMM: List of Generalized Gell-Mann Matrices.
        scramham: Scrambled Hamiltonian.
        n: Number of environmental qubits.
        arb_thetas: List of theta values that parameterize the Unitary Matrix.
        init_states: State to evolve forward.
        time: Characteristic time.

        Returns: Score.'''

    # Specfiying dimensions 
    dsys = 2
    denv = (2)**n
    D = ((dsys*denv)**2)-1

    # Multiplying arbitrary thetas to the GGMMs
    param_GGMM = []
    for i in range(0, len(arb_thetas)):  
        param_GGMM.append(arb_thetas[i]*GGMM[i])
    
    # Constructing the trial unitary matrix with the GGMMs
    unitary_trial = np.array(expm(1j*sum(param_GGMM)), dtype = complex) 

    totscores = []
    n_vecs = []
    # For each permutated matrix...
    for j in range(0, n+1):

        H_total = (unitary_trial.dot(scramhams[j])).dot(np.conjugate(np.transpose(unitary_trial)))
             
        # Computing Q alphas...
        Q_alphas = []
        for k in range(0, 3):        # For alpha = 1, 2, 3
            state = np.kron(sigma[k], env)
            Q_alphas.append(te.time_evolution(state, H_total, time))

        # Computing Matrix elements
        M = np.zeros((3,3), dtype = complex)
        for row in range(0,3):
            for col in range(0,3):
                a = np.array(red.reduce_DM_A(Q_alphas[row], 2, 2**n))
                b = np.array(red.reduce_DM_A(Q_alphas[col], 2, 2**n))
                M[row][col] = np.trace(a.dot(b))
        
        # Computing max eigenvalue and corresponding eigenvector 
        lam, V = LA.eigh(M)
        maxlam = lam[-1]
        n_vector = V[-1]

        # Adding important info into lists
        totscores.append(1/2-1/4*(maxlam))
        n_vecs.append(n_vector)

    # Computing final score 
    finalscore = sum(totscores)/len(totscores)

    return totscores, n_vecs, finalscore
    

def gradient_descent(gradient, start, learn_rate, n_iter, cost_func, adjust, adjLR):
    '''Parameters:
        gradient: Python function that takes a vector and returns the gradient of the function we want to minimize.
        start: Starting vector.
        learn_rate: Learning rate that controls magnitude of vector update
        n_iter: Number of iterations.

        Returns: Minimizing list of vectors.'''
    
    score = []
    theta_trial = start
    cost_history = np.zeros(n_iter)

    for i in range(n_iter):
        diff = -learn_rate * gradient(theta_trial)
        theta_trial += diff
        
        # # Only apply change to qubits of focus - Only use with FIXED method
        # if inds != []:
        #     for j in inds:
        #         theta_trial[j] += diff[j]
        # else:
        #     theta_trial += diff

        score_tg = cost_func(theta_trial)
        score.append(score_tg)
        #print('######## THETA GUESS ',_,' ########: \n', theta_trial)

        cost_history[i] = score_tg

        if i == adjust:
            learn_rate = adjLR
    
    # Finding min manually 
    # tmp = score[0]
    # it = 0
    # for e in score:
    #     if(e < tmp):
    #         tmp = e
    #         it += 1
    # minscore = tmp
    # thetamin = [theta_1[it], theta_2[it], theta_3[it]]
    # print('Automated find of min: ', thetamin)
    # print('Score of automated theta min: ', minscore)
    
    print('Theta final guess: ', theta_trial)

    # # Plotting
    # fig = plt.figure(num = 1, figsize=(6, 6))
    # ax = fig.add_subplot(projection='3d')

    # for i in range(len(theta_1)): #plot each point + it's index as text above
    #     ax.scatter(theta_1[i], theta_2[i], theta_3[i], 
    #        linewidths = 1, 
    #        alpha = .7,
    #        edgecolor = 'k',
    #        s = 100) 
    #     ax.text(theta_1[i], theta_2[i], theta_3[i],  '%s' % (str(i)), 
    #     size = 10, 
    #     zorder = 1,  
    #     color = 'k') 

    # return theta_trial, ax

    return theta_trial, cost_history 



# ========================= TESNOR FLOW ATTEMPTS ============================

# import tensorflow as tf

# # Method 1
# def optimize(cfunc, theta):
#     arb_theta0 = tf.Variable(theta, name = 'theta guess')
#     print(arb_theta0)
#     func = cfunc(arb_theta0)

#     optimizer = tf.train.GradientDescentOptimizer(0.5)
#     train = optimizer.minimize(func)

#     init = tf.initialize_all_variables()

#     with tf.Session() as session:
#         session.run(init)
#         print("starting at", "theta:", session.run(arb_theta0), "score:", session.run(func))
        
#     for step in range(10):
#         session.run(train)
#         print("step", step, "x:", session.run(arb_theta0), "log(x)^2:", session.run(func))

# # Method 2 
# def reset(startingtheta):	
#     for i in range(len(startingtheta)):
#         startingtheta[i] = tf.Variable(2.0) 
#     return startingtheta

# def tensorflow2(fu, fu_minimzie):
#     theta0 = reset([[0]*3])
#     print(theta0)
#     opt = tf.keras.optimizers.SGD(learning_rate=0.1)
#     for i in range(50):
#         print ('score = {:.1f}, theta1 = {:.1f}, theta2 = {:.1f}, theta3 = {:.1f}'.format(fu(theta0).numpy(), theta0[0].numpy(), theta0[1].numpy(), theta0[2].numpy()))
#         opt.minimize(fu_minimzie, var_list=[theta0[0], theta0[1], theta0[2]])