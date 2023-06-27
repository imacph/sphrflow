import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from matrix_solver import gen_rhs,matrix_solve,gen_radial_grid
from special_funcs import chebspace
from phys_quants import calc_kinetics


lmax = 200
nr = 200

freq = 1
m = 0
E = 10**-4
eta = 0.35

W_t = np.zeros(lmax+1)
W_b = np.zeros(lmax+1)

DW_t = np.zeros(lmax+1)
DW_b = np.zeros(lmax+1)

Z_t = np.zeros(lmax+1)
Z_b = np.zeros(lmax+1)

Z_t[1] = 1


A_even,A_odd = gen_rhs(W_t, W_b, DW_t, DW_b, Z_t, Z_b, lmax, nr, eta)
print(A_even,A_odd)     


W,DW,Z = matrix_solve(A_even, A_odd, freq, E, eta, lmax, nr, m)
        
np.save('W_coeffs',W)
np.save('DW_coeffs',DW)
np.save('Z_coeffs',Z)


r_i,r_o,rr = gen_radial_grid(nr, eta)
ntheta = 2*nr
theta = chebspace(0,np.pi,ntheta)

q_r,q_theta,q_phi,kin_e = calc_kinetics(W, DW, Z, eta, m, theta)

fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=200)


field = q_phi
field = np.real(field)


norm = colors.TwoSlopeNorm(vmin=np.min(field),vmax=np.max(field),vcenter=0)

#log_levs = np.logspace(np.log10(np.min(field)),np.log10(np.max(field)),200)

#norm = colors.LogNorm(np.min(field),vmax = np.max(field))
#norm = colors.SymLogNorm(vmin=np.min(field),vmax=np.max(field),linthresh=0.01)

zz = np.array([[rr[i] * np.cos(theta[j]) for j in range(ntheta)] for i in range(nr)])
xx = np.array([[rr[i] * np.sin(theta[j]) for j in range(ntheta)] for i in range(nr)])



p=ax.contourf(xx,zz,field,cmap = 'seismic',levels=200,norm=norm)
#fig.colorbar(p)

ax.plot([0,0],[-r_i,-r_o],color='k',lw=0.5,alpha=0.3)
ax.plot([0,0],[r_i,r_o],color='k',lw=0.5,alpha=0.3)
ax.plot(r_o*np.sin(theta),r_o*np.cos(theta),color='k',lw=0.5,alpha=0.3)
ax.plot(r_i*np.sin(theta),r_i*np.cos(theta),color='k',lw=0.5,alpha=0.3)
ax.plot()
ax.set_aspect('equal')
ax.axis('off')