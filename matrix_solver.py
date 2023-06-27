import numpy as np
import scipy.sparse as sparse
from tqdm import tqdm
from time import time 

# internal modules
from special_funcs import sphrharm,chebspace
from finite_differences import df1_lil,df2_lil

def gen_radial_grid(nr,eta):
    
    r_o = 1/(1-eta)
    r_i = r_o * eta
    rr = np.zeros(nr)
    if eta != 0:
        rr[1:-1] = chebspace(r_i,r_o,nr-2)
        rr[0]=r_i
        rr[-1] = r_o
        
    elif eta == 0:
        rr[:-1] = chebspace(r_i,r_o,nr-1)
        
        rr[-1] = r_o
        
    return r_i,r_o,rr

def gen_matrix(freq,E,eta,lmax,nr,m,symm):
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    
    orr = 1/rr
    ll = np.linspace(0,lmax,lmax+1)
    c = np.sqrt((ll-m+1)*(ll+m+1)/(2*ll+1)/(2*ll+3))
    I = sparse.eye(nr,format='lil')

    df1 = df1_lil(rr)
    df2 = df2_lil(rr)
    
    LoB=[[None for i in range(lmax+1)] for j in range(lmax+1)]

    orr_mul = sparse.lil_matrix(np.diag(orr),dtype=np.cfloat)
    orr_mul2 = orr_mul*orr_mul

    LoB[0][0] = I
    LoB[-1][-1] = I
    
    if symm == 0:
        
        print('generating matrices for even W')
        for l in tqdm(range(2,lmax-1,2)):
            
            lp = l * (l+1)
            L2 = lp * orr_mul2 - df2
            
            LoB[l][l] = (lp * (1j*freq*I + E*L2) -2j*m*I)* orr_mul2 * L2 
            LoB[l][l+1] = -2 * l * (l+2)*c[l] * ( df1 + (l+1) * orr_mul)*orr_mul2
            LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] * (df1 - l * orr_mul)*orr_mul2
        
        print('\ngenerating matrices for odd Z')
        for l in tqdm(range(1,lmax,2)):
            
            lp = l * (l+1)
            L2 = lp * orr_mul2 - df2
            
            LoB[l][l] = (lp * (1j*freq*I + E*L2) -2j*m*I) * orr_mul2
            LoB[l][l+1] = -2 * l * (l+2)*c[l] * ( df1 + (l+1) * orr_mul)*orr_mul2
            LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] * (df1 - l * orr_mul)*orr_mul2
            

        M = sparse.bmat(LoB,format='lil',dtype=np.cfloat)
        print('\nBlock matrix built...\n')
        
        print('Boundary conditions for W')
        for l in tqdm(range(2,lmax-1,2)):
            
            
            if eta != 0:
                M[l*nr,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr,l*nr] = 1
                
                M[l*nr+1,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr+1,l*nr+1] = 1
            
            M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-1,(l+1)*nr-1] = 1
            
            M[(l+1)*nr-2,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-2,(l+1)*nr-2] = 1
        
        print('\nBoundary conditions for Z')
        for l in tqdm(range(1,lmax,2)):
             
            if eta != 0:
                 
                M[l*nr,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr,l*nr] = 1
                 
            M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-1,(l+1)*nr-1] = 1
            
    if symm == 1:
        
        print('\ngenerating matrices for odd W')
        for l in tqdm(range(1,lmax,2)):
            
            lp = l * (l+1)
            L2 = lp * orr_mul2 - df2
            
            LoB[l][l] = (lp * (1j*freq*I + E*L2) -2j*m*I)* orr_mul2 * L2 
            LoB[l][l+1] = -2 * l * (l+2)*c[l] * ( df1 + (l+1) * orr_mul)*orr_mul2
            LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] * (df1 - l * orr_mul)*orr_mul2
        
        print('\ngenerating matrices for even Z')
        for l in tqdm(range(2,lmax-1,2)):
            
            lp = l * (l+1)
            L2 = lp * orr_mul2 - df2
            
            LoB[l][l] = (lp * (1j*freq*I + E*L2) -2j*m*I) * orr_mul2
            LoB[l][l+1] = -2 * l * (l+2)*c[l] * ( df1 + (l+1) * orr_mul)*orr_mul2
            LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] * (df1 - l * orr_mul)*orr_mul2
            

        M = sparse.bmat(LoB,format='lil',dtype=np.cfloat)
        print('\nBlock matrix built...\n')
        
        print('Boundary conditions for W')
        for l in tqdm(range(1,lmax,2)):
            
            
            if eta != 0:
                M[l*nr,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr,l*nr] = 1
                
                M[l*nr+1,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr+1,l*nr+1] = 1
            
            M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-1,(l+1)*nr-1] = 1
            
            M[(l+1)*nr-2,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-2,(l+1)*nr-2] = 1
        
        print('\nBoundary conditions for Z')
        for l in tqdm(range(2,lmax-1,2)):
             
            if eta != 0:
                 
                M[l*nr,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr,l*nr] = 1
                 
            M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-1,(l+1)*nr-1] = 1
    
    return M

def gen_rhs(W_t,W_b,DW_t,DW_b,Z_t,Z_b,lmax,nr,eta):
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    
    A_even = np.zeros((lmax+1)*nr) # Equatorially symm: W even, Z odd
    A_odd = np.zeros_like(A_even) # Equatorially anti-symm: Z even, W odd
    
    
    
    for l in range(0,lmax+1,2):
        
        A_even[l*nr] = W_b[l]
        A_even[l*nr+1] = (rr[1] - rr[0]) * DW_b[l] + W_b[l]
        
        A_even[(l+1)*nr-1] = W_t[l]
        A_even[(l+1)*nr-2] = -(rr[-1]-rr[-2]) * DW_t[l] + W_t[l]
        
        
        if l < lmax:
            
            A_even[(l+1)*nr] = Z_b[l+1]
            A_even[(l+2)*nr-1] = Z_t[l+1]
           

        A_odd[(l)*nr] = Z_b[l]
        A_odd[(l+1)*nr-1] = Z_t[l]
        
        if l < lmax:
            
            A_odd[(l+1)*nr] = W_b[l+1]
            A_odd[(l+1)*nr+1] = (rr[1] - rr[0]) * DW_b[l+1] + W_b[l+1]
            
            A_odd[(l+2)*nr-1] = W_t[l]
            A_odd[(l+2)*nr-2] = -(rr[-1]-rr[-2]) * DW_t[l+1] + W_t[l+1]
        
    return A_even,A_odd



def matrix_solve(A_even,A_odd,freq,E,eta,lmax,nr,m):
    
    Z = np.zeros((lmax+1,nr),dtype=complex)
    W = np.zeros((lmax+1,nr),dtype=complex)
    DW = np.zeros((lmax+1,nr),dtype=complex)
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    df1 = df1_lil(rr)
    
    if any([x!=0 for x in A_even]):
        
        print('There is equatorially symmetric forcing...\n')
        M = gen_matrix(freq,E,eta,lmax,nr,m,0)
        
        t0 = time()
        coeffs =  sparse.linalg.spsolve(sparse.csc_matrix(M),A_even)
        print('\nEquatorially symmetric system solved ({:.2e}s)'.format(time()-t0))
        for l in range(0,lmax-1,2):
            
            W[l,:] = coeffs[l*nr:(l+1)*nr]
            Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr]
            DW[l,:] = df1.dot(W[l,:])
            
    if any([x!=0 for x in A_odd]):
        
        print('There is equatorially anti-symmetric forcing...\n')
        M = gen_matrix(freq,E,eta,lmax,nr,m,1)
        
        coeffs =  sparse.linalg.spsolve(sparse.csc_matrix(M),A_odd)
        print('\nEquatorially symmetric system solved ({:.2e}s)'.format(time()-t0))
        
        for l in range(1,lmax,2):
            
            W[l,:] = coeffs[l*nr:(l+1)*nr]
            Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr]
            DW[l,:] = df1.dot(W[l,:])
            
    return W,DW,Z


    

    
    
    