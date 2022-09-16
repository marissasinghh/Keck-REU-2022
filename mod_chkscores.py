###################################################################################################################
###
###  Check Scores   
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  This main function of this module is to create plots of iteration vs score so that one can visualize the
###  progression of scores as the gradient descent algorithm runs
###  -- can be used to plot multiple trials against each other
###  -- can be used to plot different permutation results against each other  
###
####################################################################################################################



import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

# This module allows you to plot any number of files against each other

# Global Variables
H_num = 2
q_num = 2 # Total number of qubits

def plotfile(triosc, collet, trial_num , permyn, ham_num):

    if triosc == 'T':
      filename = open(str(trial_num) + '_systrialAVGN_CH_H' + str(H_num) + '_qu' + str(q_num) + '.txt', 'r') 
        
    else: 
      permyn = input('AVGN or EIG or CH? (A/E/C) ')
      trial_num = int(input('Which Trial '))

      if permyn == 'A':
        filename = open('systrialAVGNscores_H' + str(H_num) + '_qu' + str(q_num) + '.txt', 'r') 
      elif permyn == 'E':
        filename = open('systrialEIGscores_H' + str(H_num) + '_qu' + str(q_num) + '.txt', 'r') 
      elif permyn == 'C':
        filename = open(str(trial_num) + '_systrialAVGN_CH_H' + str(H_num) + '_qu' + str(q_num) + '.txt', 'r') 

    lines = filename.readlines()
    lenlines = int(len(lines))
    x = [0]*lenlines
    y = [0]*lenlines

    for i in range(lenlines):
        vec = literal_eval(lines[i])
        x[i] = int(vec[0])
        y[i] = float(vec[1])
        print(int(vec[0]), float(vec[1]))
    print('x', x)
    print('y', y)
    
    plots = []
    if triosc == 'T':
      plots.append(plt.scatter(x, y, marker = '*', s = 10, color = collet, label = trial_num))    # Creating a scatter plot of the data
    else:
      plots.append(plt.scatter(x, y, marker = '*', s = 10, color = collet, label = permyn)) 
    
    return plots    



# Actual code...

tos = input('Trial or Score (T/S) ')

if tos == 'T':
  t_num = int(input('Which Trial '))
  permused = input('Is perm used? (Y/NN/N) ') 
  numfiles = q_num
else:
  t_num = 0
  permused = ''
  numfiles = int(input('Enter the number of files you would like to plot '))   # User inputs how many files they wish to plot


for i in range(numfiles):  # Running plotfile() for all files inputted 
  colorl = ''
  ham_num = i
  
  if i == 0:
    colorl = 'magenta'
  elif i == 1:
    colorl = 'blue'
  elif i == 2:
    colorl = 'green'
  elif i == 3:
    colorl = 'orange'
  elif i == 4:
    colorl = 'yellow'
  elif i == 5:
    colorl = 'purple'
  elif i == 6:
    colorl = 'red'
  
  plots = plotfile(tos, colorl, t_num, permused, ham_num)

if tos == 'T':
  leg = []
  for i in range(numfiles):
    leg.append("Ham " + str(i))
else:
  leg = []
  for i in range(numfiles):
    leg.append("Trial " + str(i))

plt.legend(leg)
#plt.title("Score History (Closer to 0) for Hamiltonian %d" %H_num + ", %d" %q_num + " Qubits")
#plt.title("Cost History for Hamiltonian %d" %H_num + ", %d" %q_num + " Qubits, Trial %d" %t_num )
plt.title("Cost Histories for Hamiltonian %d" %H_num + ", %d" %q_num + " Qubits")
plt.xlabel("Iterations")
plt.ylabel("Score")
plt.show()  # Showing plots 