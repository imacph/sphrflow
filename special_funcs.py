from scipy.special import lpmv
from scipy.special import factorial
import numpy as np


def trun_fact(l,m):
    
    
    prod = 1

    for k in range(l-m+1,l+m+1):
        
        prod *= k
        
    return prod


def sphrharm(l,m,theta,phi):
    # spherical harmonics degree l order m 
    
    if m == 0:
        N = ((2*l+1)/4/np.pi )**(1/2)
    
    else:
        N = ((2*l+1)/4/np.pi / trun_fact(l,m))**(1/2)
    
    if m % 2 == 1:
        
        N*=-1 
        # this cancels the Cordon-Shortley 
        # phase factor present by default
        # in scipy assoc. Legendre funcs
    
    return N*lpmv(m,l,np.cos(theta))*np.exp(1j*m*phi)

def chebspace(start,end,num,inc=1):
    # chebyshev grid, not inclusive of endpoints
    
    if not inc:
        x = np.cos((2*np.linspace(1,num,num)-1)*np.pi/2/num)[::-1]
    
    else:
        x = np.cos(np.linspace(0,num-1,num)*np.pi/(num-1))
    
    return (end-start) * x /2 + (start+end)/2


    
    