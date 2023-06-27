import numpy as np
from scipy.sparse import lil_matrix

    
    
def df1_lil(rr):
    # first order acurate discrete first derivative, non-uniform grid
    # generated as a sparse scipy list of lists (LIL) matrix
    
    nr = len(rr)
    df1 = lil_matrix((nr,nr))
    
    for i in range(1,nr-1):
        
        df1[i,i+1] = 1/(rr[i+1]-rr[i-1])
        df1[i,i-1] = -1/(rr[i+1]-rr[i-1])
        
    return df1
        
def df2_lil(rr):
    # first order acurate discrete second derivative, non-uniform grid
    # generated as a sparse scipy list of lists (LIL) matrix
    
    nr = len(rr)
    df2 = lil_matrix((nr,nr))
    
    for i in range(1,nr-1):
        
        df2[i,i] = -2 / (rr[i+1] - rr[i]) / (rr[i] - rr[i-1])
        df2[i,i-1] = 2 / (rr[i+1] - rr[i-1]) / (rr[i] - rr[i-1])
        df2[i,i+1] = 2/ (rr[i+1] - rr[i-1]) / (rr[i+1]-rr[i])
        
    return df2