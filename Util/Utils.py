import numpy as np
import numpy.linalg as LA

def mat_sqrt(matrix):
    w, v = LA.eigh(matrix)
    
    return np.matmul(v, np.matmul(np.diag(np.sqrt(w)), v.T))