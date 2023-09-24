import numpy as np
from matrix_solver import gen_radial_grid
from special_funcs import sphrharm,chebspace

def calc_kinetics(W,DW,Z,eta,m,theta):
    
    lmax,nr = np.shape(W)
    
    lmax += -1
    
    ntheta = len(theta)
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    orr = 1/rr
    
    kin_e = np.zeros((nr,ntheta),dtype='float')

    q_r = np.zeros((nr,ntheta),dtype='complex')
    q_theta = np.zeros_like(q_r)
    q_phi = np.zeros_like(q_r)

    YY = np.zeros((lmax+2,ntheta),dtype=complex)

    for l in range(lmax+2):
        YY[l,:] = sphrharm(l,m,theta,0)
    
    ll = np.linspace(0,lmax,lmax+1)
    c = np.sqrt((ll-m+1)*(ll+m+1)/(2*ll+1)/(2*ll+3))

    for l in range(lmax+1):
            
        q_r += YY[l][np.newaxis,:] * W[l][:,np.newaxis] * l * (l+1) * orr[:,np.newaxis]**2
        
        q_theta += orr[:,np.newaxis] * DW[l][:,np.newaxis] /np.sin(theta[np.newaxis,:]) * (l * c[l] * YY[l+1][np.newaxis,:]-(l+1) * c[l-1] * YY[l-1][np.newaxis,:])


        q_phi += -Z[l][:,np.newaxis] * orr[:,np.newaxis] / np.sin(theta[np.newaxis,:]) * (l * c[l] * YY[l+1][np.newaxis,:]-(l+1) * c[l-1] * YY[l-1][np.newaxis,:])
        
    q_r = np.real(q_r)
    q_theta = np.real(q_theta)
    q_phi = np.real(q_phi)
    
    kin_e = 0.5* (q_r**2 + q_theta**2+q_phi**2)
    
    return q_r,q_theta,q_phi,kin_e

