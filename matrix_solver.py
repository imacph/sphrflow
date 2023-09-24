import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from tqdm import tqdm
from time import time 
import numpy.polynomial.chebyshev as cheb
from scipy.sparse import lil_matrix

# internal modules
from special_funcs import sphrharm,chebspace
from finite_differences import df1_lil,df2_lil,df1_sphr,df2_sphr

def gen_radial_grid(nr,eta):
    
    r_o = 1/(1-eta)
    r_i = r_o * eta
    
    if eta !=0:
        rr = chebspace(r_i,r_o,nr)[::-1]
    
    else:
        
        #rr = np.zeros(nr)
        #rr[:-1] = chebspace(r_i,r_o,nr-1,inc=0)
        
        #rr[-1] = r_o
        
        
        rr = chebspace(-1,1,2*nr)[::-1][-nr:]
        
        
        
        
    '''
    rr = np.zeros(nr)
    
    
    if eta != 0:
        rr[1:-1] = chebspace(r_i,r_o,nr-2,inc=0)
        rr[0]=r_i
        rr[-1] = r_o
        
    elif eta == 0:
        rr[:-1] = chebspace(r_i,r_o,nr-1,inc=0)
        
        rr[-1] = r_o
    
    
    '''
    
    
    
    return r_i,r_o,rr

def gen_matrix(freq,E,eta,lmax,nr,m,symm,free_slip_icb,free_slip_cmb,no_tqdm):
    
    # this routine builds the block matrix (either equatorially symmetric or anti-symmetric)
    # that solves the NS eq. in frequency space
    # the matrix alternates W,Z entries to maximize diagonality
    # can be thought of as a (lmax+1 x lmax+1) matrix with (nr x nr) matrices 
    # as entires
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    
    orr = 1/rr
    ll = np.linspace(0,lmax,lmax+1)
    
    c = np.zeros_like(ll)
    for l in range(m,lmax+1):
        c[l] = np.sqrt((l-m+1)*(l+m+1)/(2*l+1)/(2*l+3))
    I = sparse.eye(nr,format='lil')

    if eta != 0:
        df1 = df1_lil(rr)
        df2 = df2_lil(rr)
    
    if eta == 0:
        
        df1 = df1_sphr(rr,symm)
        df2 = df2_sphr(rr,symm)
    
    LoB=[[None for i in range(lmax+1)] for j in range(lmax+1)]
    
    
    
    df1 = df1_lil(rr)
    df2 = df2_lil(rr)

    
    
    
    
    '''
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
    df2 = df1 @ df1
    '''
    
    
    '''
    N = nr-1
    C_mat = lil_matrix((N+1,N+1))

    c1 = np.array([2] + [1] * (N-1) + [2])

    D_1 = lil_matrix((N+1,N+1))
    D_1[:-1,:] = cheb.chebder(np.eye(N+1),m=1,scl=-2)

    D_2 = lil_matrix((N+1,N+1))
    D_2[:-2,:] = cheb.chebder(np.eye(N+1),m=2,scl=-2)

    for i in range(N+1):
        
        for j in range(N+1):
            
            C_mat[i,j] = np.cos(i*j*np.pi/N)


    T_mat = C_mat * c1[:,np.newaxis] * c1[np.newaxis,:]
    df1 = C_mat @ D_1 @ T_mat *2/N    
    df2 = C_mat @ D_2 @ T_mat *2/N 
    '''
    
    orr_mul = sparse.lil_matrix(np.diag(orr),dtype=np.cfloat)
    orr_mul2 = orr_mul*orr_mul
    rr_mul = sparse.lil_matrix(np.diag(rr),dtype=np.cfloat)
    rr_mul2 = rr_mul @ rr_mul
    
    if m == 0:
        
        sl_odd = 1
        sl_even = 2
        
        LoB[0][0] = I
        
    elif m % 2 == 0:
        
        sl_odd = m+1
        sl_even = m 
        
    elif m % 2 == 1:
        
        sl_odd = m
        sl_even = m+1
        
    if m !=0:
        
        for l in range(1,sl_odd,2):
            
            LoB[l][l] = I
            
        for l in range(0,sl_even,2):
            
            LoB[l][l]= I
    
    LoB[-1][-1] = I
    
    if (symm == 0 and m % 2 == 0) or (symm == 1 and m %2 == 1):
        
        if eta != 0:
            if not no_tqdm: print('generating matrices for even W')
            for l in tqdm(range(sl_even,lmax-1,2),disable=no_tqdm):
                
                
                lp = l * (l+1)
                L2 = lp * orr_mul2 - df2
                
                LoB[l][l] =orr_mul2* (lp * (1j*freq*I + E*L2) -2j*m*I) * L2 
                LoB[l][l+1] = -2 * l * (l+2)*c[l] *orr_mul2* ( df1 + (l+1) * orr_mul)
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] *orr_mul2*(df1 - l * orr_mul)
            
            if not no_tqdm: print('\ngenerating matrices for odd Z')
            for l in tqdm(range(sl_odd,lmax,2),disable=no_tqdm):
                
                lp = l * (l+1)
                L2 = lp * orr_mul2 - df2
                
                LoB[l][l] = orr_mul2*(lp * (1j*freq*I + E*L2) -2j*m*I) 
                LoB[l][l+1] = -2 * l * (l+2)*c[l] * orr_mul2* ( df1 + (l+1) * orr_mul)
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] * orr_mul2*(df1 - l * orr_mul)
            
        if eta == 0:
            if not no_tqdm: print('generating matrices for even W')
            for l in tqdm(range(sl_even,lmax-1,2),disable=no_tqdm):
                
                lp = l * (l+1)
                L2 = lp * orr_mul2   - df2 
                
                LoB[l][l] = orr_mul2*(lp * (1j*freq*I + E*L2) -2j*m*I) * L2 * rr_mul2
                LoB[l][l+1] = -2 * l * (l+2)*c[l] * orr_mul2*( df1  + (l+1) * orr_mul)* rr_mul
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] * orr_mul2*(df1 - l * orr_mul)* rr_mul
                
        
            if not no_tqdm: print('\ngenerating matrices for odd Z')
            for l in tqdm(range(sl_odd,lmax,2),disable=no_tqdm):
                
                lp = l * (l+1)
                L2 = lp * orr_mul2 - df2
                
                LoB[l][l] =orr_mul2* (lp * (1j*freq*I + E*L2) -2j*m*I) * rr_mul
                LoB[l][l+1] = -2 * l * (l+2)*c[l] * orr_mul2*( df1 + (l+1) * orr_mul) * rr_mul2
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] * orr_mul2*(df1 - l * orr_mul) * rr_mul2
        
        M = sparse.bmat(LoB,format='lil',dtype=np.cfloat)
        if not no_tqdm: print('\nBlock matrix built...\n')
        
        if not no_tqdm: print('Boundary conditions for W')
        for l in tqdm(range(sl_even,lmax-1,2),disable=no_tqdm):
            
            # Non-penetration condition at ICB
            M[l*nr,(l-1)*nr:(l+2)*nr] = 0
            M[l*nr,l*nr] = 1
            
            # Slip conditions at ICB
            if free_slip_icb:
                
                M[l*nr+1,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr+1,l*nr:(l+1)*nr] = df2[1,:] - 2 / rr[1] * df1[1,:]
                
            else:
                
                M[l*nr+1,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr+1,l*nr+1] = 1
            
            # Non-penetration at CMB
            M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-1,(l+1)*nr-1] = 1
            
            # Slip conditions at CMB
            if free_slip_cmb:
                
                M[(l+1)*nr-2,(l-1)*nr:(l+2)*nr] = 0
                M[(l+1)*nr-2,l*nr:(l+1)*nr] = df2[-2,:] - 2 / rr[-2] * df1[-2,:]
                
            else:
                
                M[(l+1)*nr-2,(l-1)*nr:(l+2)*nr] = 0
                M[(l+1)*nr-2,(l+1)*nr-2] = 1
            
        if not no_tqdm: print('\nBoundary conditions for Z')
        for l in tqdm(range(sl_odd,lmax,2),disable=no_tqdm):
            
            
                  
            if free_slip_icb:
                
                M[l*nr,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr,l*nr:(l+1)*nr] = df1[0,:] - 2 / rr[0]*I[0,:]

            else:
                M[l*nr,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr,l*nr] = 1
                  
            if free_slip_cmb:
                
                M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
                M[(l+1)*nr-1,l*nr:(l+1)*nr] = df1[-2,:] - 2 / rr[-2]*I[-2,:]
            
            else:
                 M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
                 M[(l+1)*nr-1,(l+1)*nr-1] = 1
             
            
    if (symm == 1 and m % 2 == 0) or (symm == 0 and m %2 == 1):
        
        if eta != 0:
        
            if not no_tqdm: print('\ngenerating matrices for odd W')
            for l in tqdm(range(sl_odd,lmax,2),disable=no_tqdm):
                
                lp = l* (l+1)
                L2 = lp * orr_mul2 - df2
                
                LoB[l][l] =  orr_mul2*(lp * (1j*freq*I + E*L2) -2j*m*I) * L2 
                LoB[l][l+1] = -2 * l * (l+2)*c[l] *orr_mul2 *( df1 + (l+1) * orr_mul)
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] *orr_mul2* (df1 - l * orr_mul)
            
            if not no_tqdm: print('\ngenerating matrices for even Z')
            for l in tqdm(range(sl_even,lmax-1,2),disable=no_tqdm):
                
                lp = l * (l+1)
                L2 = lp * orr_mul2 - df2
                
                LoB[l][l] =orr_mul2*(lp * (1j*freq*I + E*L2) -2j*m*I) 
                LoB[l][l+1] = -2 * l * (l+2)*c[l] *orr_mul2* ( df1 + (l+1) * orr_mul)
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] *orr_mul2* (df1 - l * orr_mul)
        
        if eta == 0:
            
            if not no_tqdm: print('\ngenerating matrices for odd W')
            for l in tqdm(range(sl_odd,lmax,2),disable=no_tqdm):
                
                lp = l* (l+1)
                L2 = lp * orr_mul2 - df2
                
                LoB[l][l] =  orr_mul2*(lp * (1j*freq*I + E*L2) -2j*m*I) * L2 * rr_mul
                LoB[l][l+1] = -2 * l * (l+2)*c[l] *orr_mul2 *( df1 + (l+1) * orr_mul) * rr_mul2
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] *orr_mul2* (df1 - l * orr_mul) * rr_mul2
            
            if not no_tqdm: print('\ngenerating matrices for even Z')
            for l in tqdm(range(sl_even,lmax-1,2),disable=no_tqdm):
                
                lp = l * (l+1)
                L2 = lp * orr_mul2 - df2
                
                LoB[l][l] =orr_mul2*(lp * (1j*freq*I + E*L2) -2j*m*I) * rr_mul2
                LoB[l][l+1] = -2 * l * (l+2)*c[l] *orr_mul2* ( df1 + (l+1) * orr_mul) * rr_mul
                LoB[l][l-1] = -2 * (l-1) * (l+1) * c[l-1] *orr_mul2* (df1 - l * orr_mul) * rr_mul

        M = sparse.bmat(LoB,format='lil',dtype=np.cfloat)
        if not no_tqdm: print('\nBlock matrix built...\n')
        
        if not no_tqdm: print('Boundary conditions for W')
        for l in tqdm(range(sl_odd,lmax,2),disable=no_tqdm):
            
            M[l*nr,(l-1)*nr:(l+2)*nr] = 0
            M[l*nr,l*nr] = 1
            
            if free_slip_icb:
                
                M[l*nr+1,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr+1,l*nr:(l+1)*nr] = df2[1,:] - 2 / rr[1] * df1[1,:]
            
            else:
                
                M[l*nr+1,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr+1,l*nr+1] = 1
            
            
            M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
            M[(l+1)*nr-1,(l+1)*nr-1] = 1
            
            if free_slip_cmb:
                
                M[(l+1)*nr-2,(l-1)*nr:(l+2)*nr] = 0
                M[(l+1)*nr-2,l*nr:(l+1)*nr] = df2[-2,:] - 2 / rr[-2] * df1[-2,:]
                
            else:
                
                M[(l+1)*nr-2,(l-1)*nr:(l+2)*nr] = 0
                M[(l+1)*nr-2,l*nr:(l+1)*nr] = df1[-1,:]
        
        if not no_tqdm: print('\nBoundary conditions for Z\n')
        for l in tqdm(range(sl_even,lmax-1,2),disable=no_tqdm):
             
            if free_slip_icb:
                M[l*nr+1,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr+1,l*nr:(l+1)*nr] = df1[1,:] - 2 / rr[1]*I[1,:]
            else:
                M[l*nr,(l-1)*nr:(l+2)*nr] = 0
                M[l*nr,l*nr] = 1
                  
            if free_slip_cmb:
                
                M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
                M[(l+1)*nr-1,l*nr:(l+1)*nr] = df1[-2,:] - 2 / rr[-2]*I[-2,:]
            
            else:
                 M[(l+1)*nr-1,(l-1)*nr:(l+2)*nr] = 0
                 M[(l+1)*nr-1,(l+1)*nr-1] = 1
    
    return M

def gen_rhs(W_t,W_b,DW_t,DW_b,Z_t,Z_b,lmax,nr,eta,m):
    
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
           

        #A_even[l*nr:(l+1)*nr] = A_even[l*nr:(l+1)*nr][::-1]
        #A_even[(l+1)*nr:(l+2)*nr] = A_even[(l+1)*nr:(l+2)*nr][::-1]
        
        A_odd[(l)*nr] = Z_b[l]
        A_odd[(l+1)*nr-1] = Z_t[l]
        
        if l < lmax:
            
            A_odd[(l+1)*nr] = W_b[l+1]
            A_odd[(l+1)*nr+1] = (rr[1] - rr[0]) * DW_b[l+1] + W_b[l+1]
            
            A_odd[(l+2)*nr-1] = W_t[l+1]
            A_odd[(l+2)*nr-2] = -(rr[-1]-rr[-2]) * DW_t[l+1] + W_t[l+1]
    
        
    
    if m % 2 == 1:
        
        A_odd,A_even = A_even,A_odd
    
    return A_even,A_odd



def matrix_solve(A_even,A_odd,freq,E,eta,lmax,nr,m,free_slip_icb =0,free_slip_cmb=0,no_tqdm=0):
    
    Z = np.zeros((lmax+1,nr),dtype=complex)
    W = np.zeros((lmax+1,nr),dtype=complex)
    DW = np.zeros((lmax+1,nr),dtype=complex)
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    
    
    
    
    
    df1 = df1_lil(rr)

    if any([x!=0 for x in A_even]):
        
        if not no_tqdm: print('There is equatorially symmetric forcing...\n')
        M = gen_matrix(freq,E,eta,lmax,nr,m,0,free_slip_icb,free_slip_cmb,no_tqdm)
        
        if not no_tqdm: print('\nSolving matrix system...')
        t0 = time()
        coeffs =  scipy.sparse.linalg.spsolve(sparse.csc_matrix(M),A_even)
        
        
        
        if not no_tqdm: print('\nEquatorially symmetric system solved ({:.2e}s)'.format(time()-t0))
        
        
        if eta !=0:
        
            if m % 2 == 0:
                for l in range(0,lmax-1,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr]
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr]
                    DW[l,:] = df1.dot(W[l,:])
                    
            if m % 2 == 1:
                for l in range(1,lmax,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr]
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr]
                    DW[l,:] = df1.dot(W[l,:])
                    
        if eta == 0:
            
            rr_mul = sparse.lil_matrix(np.diag(rr),dtype=np.cfloat)
            rr_mul2 = rr_mul @ rr_mul
            
            if m % 2 == 0:
                for l in range(0,lmax-1,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr] * rr_mul2
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr] * rr_mul
                    DW[l,:] = df1.dot(W[l,:])
                    
            if m % 2 == 1:
                for l in range(1,lmax,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr] * rr_mul
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr] * rr_mul2
                    DW[l,:] = df1.dot(W[l,:])
            
    if any([x!=0 for x in A_odd]):
        
        if not no_tqdm: print('There is equatorially anti-symmetric forcing...\n')
        M = gen_matrix(freq,E,eta,lmax,nr,m,1,free_slip_icb,free_slip_cmb,no_tqdm)
        
        if not no_tqdm: print('\nSolving matrix system...')
        t0 = time()
        coeffs =  scipy.sparse.linalg.spsolve(sparse.csc_matrix(M),A_odd)
        if not no_tqdm: print('\nEquatorially anti-symmetric system solved ({:.2e}s)'.format(time()-t0))
        
        
        if eta != 0:
            if m % 2 == 0:
                for l in range(1,lmax,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr]
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr]
                    DW[l,:] = df1.dot(W[l,:])
            if m % 2 == 1:
                for l in range(0,lmax-1,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr]
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr]
                    DW[l,:] = df1.dot(W[l,:])
                    
        if eta == 0:
            
            rr_mul = sparse.lil_matrix(np.diag(rr),dtype=np.cfloat)
            rr_mul2 = rr_mul @ rr_mul
            
            if m % 2 == 0:
                for l in range(1,lmax,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr] * rr_mul
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr] * rr_mul2
                    DW[l,:] = df1.dot(W[l,:])
                    
            if m % 2 == 1:
                for l in range(0,lmax-1,2):
                    
                    W[l,:] = coeffs[l*nr:(l+1)*nr] * rr_mul2
                    Z[l+1,:] = coeffs[(l+1)*nr:(l+2)*nr] * rr_mul
                    DW[l,:] = df1.dot(W[l,:])
            
                
    return W,DW,Z
    

    

    
    
    