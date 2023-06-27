from scipy.special import lpmv
from scipy.special import factorial
import numpy as np

def sphrharm(l,m,theta,phi):
    
    if m == 0:
        N = ((2*l+1)/4/np.pi )**(1/2)
    
    else:
        N = ((2*l+1)/4/np.pi * factorial(l-m)/factorial(l+m))**(1/2)
    
    if m % 2 == 1:
        
        N*=-1
    
    return N*lpmv(m,l,np.cos(theta))*np.exp(1j*m*phi)

def chebspace(start,end,num):
    
    x = np.cos((2*np.linspace(1,num,num)-1)*np.pi/2/num)[::-1]
    
    return (end-start) * x /2 + (start+end)/2