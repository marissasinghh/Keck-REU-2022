###################################################################################################################
###
###  Running Code 
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this file is to grab from all main modules and continually run algorithms 
###  in terms of trials
###  -- everything can be adjusted here at the top 
###  -- to run all code: run runsysfile 
###
####################################################################################################################


import mod_hamilt1 as H1
import mod_hamilt2 as H2
import mod_hamilt3 as H3
import mod_hamilt4 as H4
import mod_GGMM as gg
import run_sys_eig as re
import run_sys_gd as rsg
import run_gd_unfixed as rgu


# Input quick changes...
runnum = 2         # Number of times to run
nqs = 1            # Number of env qubits
hamnum = 2         # Hamiltonian number
cc = 1             # Coupling coefficient
its = 50           # Iterations for gradient descent 
lrate = 0.5        # Learning Rate -- 2.3 for 6+ qubits // 0.6 for 2 qubits  
adj = 25           # Adjustment iteration 
newlrate = 0.3     # Adjusted learning rate

# H2QU2 -- lrate:0.6 its:35, adj:25, newlrate:0.3
# H4QU2 -- lrate:0.5 its:35, adj:25, newlrate:0.3
# H2QU3 -- lrate:2.3 its:80, adj:60, newlrate:0.6???

### DEFINING GLOBAL VARIABLES ###

# Specifying Dimension 
d_sys = 2
d_env = 2**nqs

# Constructing GGMMs
sysGGMMs = gg.construct_allsys_ggmm(d_sys,d_env) 
ggmmlabels, allGGMMs = gg.construct_all_ggmm_nm(d_sys, nqs)
#allGGMMs = gg.construct_all_ggmm(d_sys, d_env)

# Unscrambled Hamiltonian
if hamnum == 1:
    Htot = H1.construct_H1(nqs, cc) 
elif hamnum == 2:
    Htot = H2.construct_H2(nqs, cc)
elif hamnum == 3:
    Htot = H3.construct_H3(nqs, cc)
elif hamnum == 4:
    Htot = H4.construct_H4(nqs, cc)

# Run for matrix method 
trial_num = 50
for i in range(runnum):
    print("############# Trial ", trial_num, " #############")
    
    # Run for gradient descent method 
    print("------------- GRADIENT DESCENT OUTPUT -------------")
    H_check = rgu.runsgd(trial_num, its, lrate, hamnum, nqs, sysGGMMs, Htot, allGGMMs, ggmmlabels, adj, newlrate)


    # Run for matrix method 
    print("------------- MATRIX METHOD OUTPUT -------------")
    re.runse(trial_num, hamnum, nqs, sysGGMMs, Htot, allGGMMs, H_check)
    trial_num += 1
