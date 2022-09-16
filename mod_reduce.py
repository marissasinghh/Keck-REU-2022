###################################################################################################################
###
###  Reduce Matrix
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  The main function of this module is to reduce a given matrix over a specified basis 
###  -- takes in three variables:
###  ----> matrix: matrix to be reduced
###  ----> dim_a: dimensions of system A 
###  ----> dim_b: dimensions of system B
###  ----> sys: system that you are reducing (either 'A' or 'B')
###  -- outputs reduced density matrix 
###
####################################################################################################################


import numpy as np


# =============================== NEW CODE ===============================

def reduce_DM_A(matrix, d_sys, d_env):
    '''Parameters:
        matrix: Matrix to take trace of.
        d_sys: Dimensions of system.
        d_env: Dimensions of environment.
        
    Returns: Reduced density matrix of sys.'''

    dm_vals = []
    val = 0

    reduced_density = np.zeros((d_sys, d_sys), dtype=complex)
    for row in range(0, d_env*2, d_env): # iteration down matrix
        #print("Row", row)
        for col in range(0, d_env*2, d_env): # iteration across matrix
            #print("Col", col)
            for k in range(d_env): # iteration that sums diagonals 
                val += matrix[row+k][col+k]
                #print("Diagonal", k)
                #print(val)
            dm_vals.append(val)
            val = 0
    
    reduced_density = [dm_vals[i:i+d_sys] for i in range(0, len(dm_vals), d_sys)]

    return reduced_density


# =============================== OLD CODE ===============================

def translate_indices(dim_a, dim_b):
    '''Converts Hilbert space dimensions to density matrix index convention.
        
        Parameters:
        n: dimension of hilbert spaces.
        
        Returns: dictionary of dimensions mapped to indices.'''

    indices = {}
    x = 0
    for i in range(0, dim_a):
        for j in range(0, dim_b):
            indices[i, j] = x
            x += 1
    return indices
        
### ONLY WORKS FOR SYSTEMS THAT HAVE DIMENSION 2X2
def construct_reduced_density(matrix, dim_a, dim_b, sys):
    '''Takes the reduced trace over system b if sys = 'A' 
       OR takes the reduced trace over system a if sys = 'B'
    
    Parameters:
        matrix: Matrix to take trace of.
        dim_a: Dimensions of system a.
        dim_b: Dimensions of system b.
        
        Returns: Reduced density matrix of sys.
    '''
    indices = translate_indices(dim_a, dim_b)
    reduced_density = np.zeros((dim_a, dim_b), dtype=complex)

    if sys == 'A':
        for row in range(dim_a):
            for col in range(dim_b):
                val = 0
                for i in range(dim_b):
                    val +=  matrix[indices[(row, i)]][indices[(col, i)]]
                reduced_density[row][col] = val

    elif sys == 'B':
        for row in range(dim_b):
            for col in range(dim_a):
                val = 0
                for i in range(dim_a):
                    val +=  matrix[indices[(i, row)]][indices[(i, col)]]
                reduced_density[row][col] = val
    return reduced_density
 

# =============================== TESTS REDUCED DENSITY ===============================

# Test confirming partial trace for 
# [[1, 2, 3, 4],
#  [1, 2, 3, 4],
#  [1, 2, 3, 4],
#  [1, 2, 3, 4]]
def test_rd_1():
    matrix = [[j for j in range(1,5)] for i in range(4)]
    result_a =[[3,7], [3,7]]
    result_b = [[4,6], [4,6]]
    #print(construct_reduced_density(matrix, 2, 2, 'A') == result_a)
    #print(construct_reduced_density(matrix, 2, 2, 'B') == result_b)
    print("OG: ", construct_reduced_density(matrix, 2, 2, 'A'))
    print("NEW: ", reduce_DM_A(matrix, 2, 2))

# Test confirming partial trace for
# [[1, 0, 0, 0],
#  [0, 0, 0, 0]
#  [0, 0, 0, 0]
#  [0, 0, 0, 0]]
def test_rd_2():
    matrix = np.zeros((4,4), int)
    matrix[0][0] = 1
    result_a = [[1,0], [0,0]]
    result_b = [[1,0], [0,0]]
    #print(construct_reduced_density(matrix, 2, 2, 'A'))
    #print(construct_reduced_density(matrix, 2, 2, 'B'))
    print("OG: ", construct_reduced_density(matrix, 2, 2, 'A'))
    print("NEW: ", reduce_DM_A(matrix, 2, 2))


#Test confirming partial trace  for
#[[1/2, 0, -1/2, 0]
# [0,   0,   0,  0]
# [1/2, 0,  1/2, 0]
# [0,   0,   0,  0]]
def test_rd_3():
    matrix = [[1/2, 0, -1/2, 0], 
                 [0,0,0,0], 
               [1/2,0,1/2,0],
                 [0,0,0,0]]
    result_a = [[1/2,-1/2], [1/2,1/2]]
    result_b = [[1,0], [0,0]]
    #print(construct_reduced_density(matrix, 2, 2, 'A'))
    #print(construct_reduced_density(matrix, 2, 2, 'B'))
    print("OG: ", construct_reduced_density(matrix, 2, 2, 'A'))
    print("NEW: ", reduce_DM_A(matrix, 2, 2))


#Test confirming partial trace  for
#[[1, 2],             [[-2, 4],
# [3,4]]     tensor    [9, 7]]
def test_rd_4():
    matrix1 = [[1,2], [3,4]]
    matrix2 = [[-2, 4], [9,7]]
    result = np.kron(matrix1, matrix2)
    print(result)
    #print(construct_reduced_density(result, 2, 2, 'A'))
    #print(construct_reduced_density(result, 2, 2, 'B'))
    print("OG: ", construct_reduced_density(result, 2, 2, 'A'))
    print("NEW: ", reduce_DM_A(result, 2, 2))


def test_rd_5():
    matrix = [[1, 2, 3, 4, 2, 2, 3, 4], 
              [1, 2, 3, 4, 2, 2, 3, 4], 
              [5, 6, 7, 8, 5, 6, 7, 8],
              [5, 6, 7, 8, 5, 6, 7, 8],
              [3, 2, 3, 4, 4, 2, 3, 4], 
              [1, 2, 3, 4, 1, 2, 3, 4], 
              [5, 6, 7, 8, 5, 6, 7, 8],
              [5, 6, 7, 8, 5, 6, 7, 8]]
    print(matrix)
    result_a = [[18, 19], [20, 21]]
    #print(construct_reduced_density(matrix, 2, 2, 'A'))
    #print(construct_reduced_density(matrix, 2, 2, 'B'))
    #print("OG: ", construct_reduced_density(matrix, 4, 4, 'A'))
    print("NEW: ", reduce_DM_A(matrix, 2, 4))


# Test
M = [[1/2, 0, 0, 1/2], [0, 0, 0, 0], [0, 0, 0, 0], [1/2, 0, 0, 1/2]]
# print("ORIGINAL: ", construct_reduced_density(M, 2, 2, 'A'))
# print("NEW: ", reduce_DM_A(M, 2, 2))

# test_rd_1()
# test_rd_2()
# test_rd_3()
# test_rd_4()
# test_rd_5()
