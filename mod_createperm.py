###################################################################################################################
###
###  Creating Permutation Matrices
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to create a permutation matrix that will switch the role of "system"
###  to each environmental qubit
###  -- takes in two variables:
###  ----> n: number of total qubits
###  ----> slot: number of environmental qubit that you want to switch with "system"
###  -- outputs the appropriate permutation matrix
###
####################################################################################################################


import numpy as np

def decimalToBinary(num, n):
    '''This function returns a binary number with the appropriate amount of leading 0s.
       
       Parameters:
       num: number to be converted to binary.
       n: expected length of the binary number. 
       
       Returns: Single binary number.'''

    bin = "{0:b}".format(int(num))
    if len(bin) == n:
        return "{0:b}".format(int(num))
    else:
        return str(bin).zfill(n)
        

def createperm(n, slot):
    '''This function returns a permutation matrix that switches slots in a tensor product.
       
       Parameters:
       n: number of total qubits.
       slot: slot to be switched with slot A. 
       
       Returns: One permutation matrix.'''
    
    slot = slot - 1
    perm = np.zeros((2**n, 2**n))

    # Creating lists of binary numbers to represent states of the qubits
    binlist = []
    for i in range(2**n):
        binlist.append(str(decimalToBinary(i, n)))

    # Iterating through lists to see which are the same 
    for i in range(len(binlist)):
        
        if binlist[i][0] == binlist[i][slot]:
            perm[i][i] = 1
        
        else:
            # First creating the number that we want to switch to 
            numsearch = ''
            for j in range(len(binlist[i])):
                if j == 0:
                    numsearch += binlist[i][slot]
                elif j == slot:
                    numsearch += binlist[i][0]
                else:
                    numsearch += binlist[i][j]
            
            # Finding where that number lays in the binlist
            k = 0 
            while binlist[k] != numsearch:
                k += 1
            
            # Changing the appropriate slot in the perm matrix
            perm[k][i] = 1 
    
    return(perm)


## Testing ##
# createperm(2, 2)
# createperm(3, 2)
# createperm(3, 3)