import numpy as np
from scipy.sparse import lil_matrix
from special_funcs import chebspace
    
    
def df1_lil(rr,cheb=False):
    # first order acurate discrete first derivative, non-uniform grid
    # generated as a sparse scipy list of lists (LIL) matrix to conserve memory
    
    nr = len(rr)
    
    if not cheb:
        df1 = lil_matrix((nr,nr))
        
        
        df1[0,0] = -1/(rr[1]-rr[0])
        df1[0,1] = 1/(rr[1]-rr[0])
        
        df1[-1,-1] = 1 / (rr[-1] - rr[-2])
        df1[-1,-2] = -1 / (rr[-1] - rr[-2])
        
        for i in range(1,nr-1):
            
            df1[i,i+1] = 1/(rr[i+1]-rr[i-1])
            df1[i,i-1] = -1/(rr[i+1]-rr[i-1])
            
        return df1
    if cheb:   
        if np.array_equal(chebspace(rr[0],rr[-1],nr)[::-1], rr):
        
            N = nr - 1
            
            df1 = lil_matrix((N+1,N+1))
            xx = chebspace(-1,1,nr)
            
            c1 = np.array([2] + [1] * (N-1) + [2])
            
            
            
            for i in range(N+1):
                
                for j in range(N+1):
                    
                    if i == j:
                        
                        if i == 0:
                            df1[i,i] = (2* N**2+1)/3
                        elif i == N:
                            df1[i,i] = - (2* N**2+1)/3
                        else:
                            df1[i,j] = - xx[i] / (1-xx[i]**2)
                    
                    else:
                        
                        df1[i,j] = 2*c1[i]/c1[j] * (-1)**(i+j)/(xx[i] - xx[j])
                  
            df1[:,:] = df1[::-1,::-1]
        
        else:
            
            N = 2*nr - 1
            
            df1 = lil_matrix((N+1,N+1))
            xx = chebspace(-1,1,2*nr)
            
            
            
            c1 = np.array([2] + [1] * (N-1) + [2])
            
            
            
            for i in range(N+1):
                
                for j in range(N+1):
                    
                    if i == j:
                        
                        if i == 0:
                            df1[i,i] = (2* N**2+1)/6
                        elif i == N:
                            df1[i,i] = - (2* N**2+1)/6
                        else:
                            df1[i,j] = - xx[i] / (1-xx[i]**2)/2 
                    
                    else:
                        
                        df1[i,j] = c1[i]/c1[j] * (-1)**(i+j)/(xx[i] - xx[j])
                  
            df1 = df1[::-1,::-1][-nr:,-nr:]
            
            
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
        
    df1 = df1_lil(rr)
    
    df2 = df1.dot(df1)
        
    return df2

def df1_sphr(rr,symm):
    
    # first order acurate discrete first derivative, non-uniform grid
    # generated as a sparse scipy list of lists (LIL) matrix to conserve memory
    
    nr = len(rr)
    df1 = lil_matrix((nr,nr))
    
    if symm == 0:
        
        df1[0,0] = -1/(rr[1]-rr[0])/2
        
    else:
        
        df1[0,0] = 1/(rr[1]-rr[0]) / 2
        
    df1[0,1] = 1/(rr[1]-rr[0])/2
    
    df1[-1,-1] = 1 / (rr[-1] - rr[-2])
    df1[-1,-2] = -1 / (rr[-1] - rr[-2])
    
    for i in range(1,nr-1):
        
        df1[i,i+1] = 1/(rr[i+1]-rr[i-1])
        df1[i,i-1] = -1/(rr[i+1]-rr[i-1])
        
    return df1

def df2_sphr(rr,symm):

    df1 = df1_sphr(rr,symm)
    
    df2 = df1.dot(df1)
    
    if symm == 0:
        
        df2[0,0] = -1/(rr[1]-rr[0])**2
        df2[0,1] = 1/(rr[1]-rr[0])**2
    
    if symm == 1:
        
        df2[0,0] = -3/(rr[1]-rr[0])**2
        df2[0,1] = 1/(rr[1]-rr[0])**2
        
    
    return df2