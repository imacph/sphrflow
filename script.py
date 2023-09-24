import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

from matrix_solver import gen_rhs,matrix_solve,gen_radial_grid
from special_funcs import chebspace,sphrharm
from phys_quants import calc_kinetics,calc_visc_stress,calc_kin_energy,calc_advection
from finite_differences import df1_lil 

re = np.real
im = np.imag

def meridional_slice(field,rr,theta,crit_lat_freq,scale='symlog',step=2,figax=None):
    
    
    if figax == None: 
        fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=800)
    else: 
        fig,ax=figax
    
    ntheta = len(theta)
    nr = len(rr)
    
    zz = np.array([[rr[i] * np.cos(theta[j]) for j in range(0,ntheta,step)] for i in range(0,nr,step)])
    xx = np.array([[rr[i] * np.sin(theta[j]) for j in range(0,ntheta,step)] for i in range(0,nr,step)])
    
    vmin = np.min(field)
    vmax = np.max(field)


    if -vmin > vmax:
        
        vmax1 = -vmin
        vmin1 = vmin
        
    elif -vmin <= vmax:
        
        vmin1 = -vmax
        vmax1 = vmax
    
    
        
    min_dec = int(np.ceil(np.log10(-vmin1)))
    max_dec = int(np.ceil(np.log10(vmax1)))

    thres = max(abs(vmin),abs(vmax))/15
    
    norm = colors.SymLogNorm(vmin = vmin1, vmax = vmax1, linthresh=thres)
    levels = -np.logspace(np.log10(thres),min_dec,100)[::-1]
    levels = np.append(levels,np.linspace(-thres,thres,100)[1:])
    levels = np.append(levels,np.logspace(np.log10(thres),max_dec,100)[1:])
    p=ax.contourf(xx,zz,field[::step,::step],cmap = 'seismic',levels=levels,norm=norm)
    cbar = fig.colorbar(p)
    cbar.set_ticks([vmin,0,vmax])
    cbar.ax.set_yticklabels([round(vmin,15),0,round(vmax,15)])
    
    ax.set_aspect('equal')
    ax.axis('off')
       
    tick_w = 0.025
    
    crit_lat = np.arccos(crit_lat_freq/2)
    
    rr_tick = np.linspace(rr[-1]-tick_w,rr[-1]+tick_w,10)
    
    ax.plot(rr_tick*np.sin(crit_lat),rr_tick*np.cos(crit_lat),color='k',lw=1)
    ax.plot(rr_tick*np.sin(crit_lat),-rr_tick*np.cos(crit_lat),color='k',lw=1)



def lib_homogeneous_prob(lmax,nr,freq,eta,E,Re):
    
    W_t = np.zeros(lmax+1)
    W_b = np.zeros(lmax+1)

    DW_t = np.zeros(lmax+1)
    DW_b = np.zeros(lmax+1)

    Z_t = np.zeros(lmax+1)
    Z_b = np.zeros(lmax+1)

    Z_t[1] = 2*np.sqrt(np.pi/3)/(1-eta)*np.sqrt(E) * Re
    
    A_even,A_odd = gen_rhs(W_t, W_b, DW_t, DW_b, Z_t, Z_b, lmax, nr, eta,0)   

    W,DW,Z = matrix_solve(A_even, A_odd, freq, E, eta, lmax, nr, 0,free_slip_icb=1,free_slip_cmb=0,no_tqdm=1)
    
    return W,DW,Z




def lib_first_order_mean(W,DW,Z,eta,E,Re):
    
    
    lmax,nr = np.shape(W)
    lmax += -1
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    ntheta = 2*nr
    theta = chebspace(0,np.pi,ntheta,inc=0)
    adv_r_re,adv_r_im,adv_theta_re,adv_theta_im,adv_phi_re,adv_phi_im = calc_advection(W,DW,Z,eta,0,theta,freq,E)

    A_r = np.real((adv_r_re+adv_r_im)/2)
    A_theta = np.real((adv_theta_re+adv_theta_im)/2)
    A_phi = np.real((adv_phi_re+adv_phi_im)/2)

    df1 = df1_lil(rr)

    A_theta_int = np.zeros_like(A_phi)
    A_phi_int = np.zeros_like(A_phi)
    for i in range(ntheta):
        
        A_phi_int[:,i] = np.trapz(A_phi[:,:i+1],x=theta[:i+1])
        A_theta_int[:,i] = np.trapz(A_theta[:,:i+1],x=theta[:i+1])
    YY = np.zeros((lmax+1,ntheta),dtype=float)

    for l in range(0,lmax+1):
        
        YY[l,:] = np.real(sphrharm(l,0,theta,0))



    Z_A = np.zeros((lmax+1,nr))
    W_A = np.zeros((lmax+1,nr))
    DW_A = np.zeros((lmax+1,nr))
    for l in range(0,lmax+1):
        
        Z_A[l,:] = -2*np.pi*rr*np.trapz(A_phi_int*np.sin(theta)*YY[l,:],x=theta)
        
        DW_A[l,:] = 2*np.pi*rr*np.trapz(A_theta_int*np.sin(theta)*YY[l,:],x=theta)
        
        for i in range(nr):
            
            W_A[l,i] = np.trapz(DW_A[l,:i+1],x=rr[:i+1])


        

    A_even = np.zeros((lmax+1)*nr) # Equatorially symm: W even, Z odd
    A_odd = np.zeros_like(A_even) # Equatorially anti-symm: Z even, W odd

    for l in range(1,lmax):
        
        if l % 2 == 0:
            
            A_even[l*nr:(l+1)*nr] = -l*(l+1)/rr**2 * (l*(l+1)/rr**2 * W_A[l,:] - df1.dot(df1.dot(W_A[l,:])))
            A_odd[l*nr:(l+1)*nr] = -l*(l+1)/rr**2 * Z_A[l,:] 
            
            A_even[l*nr] = 0
            A_even[l*nr+1] = 0
            A_even[(l+1)*nr-1] = 0
            A_even[(l+1)*nr-2] = 0
            
            A_odd[l*nr] = 0
            A_odd[l*nr-1] = 0
            
        if l % 2 == 1:
            
            A_even[l*nr:(l+1)*nr] = -l*(l+1)/rr**2 * Z_A[l,:] 
            A_odd[l*nr:(l+1)*nr] = -l*(l+1)/rr**2 * (l*(l+1)/rr**2 * W_A[l,:] - df1.dot(df1.dot(W_A[l,:])))
            
            A_odd[l*nr] = 0
            A_odd[l*nr+1] = 0
            A_odd[(l+1)*nr-1] = 0
            A_odd[(l+1)*nr-2] = 0
            
            A_even[l*nr] = 0
            A_even[l*nr-1] = 0
            
            
    W,DW,Z = matrix_solve(A_even, A_odd, 0., E, eta, lmax, nr, 0,free_slip_icb=0,free_slip_cmb=0)
    
    return W,DW,Z,A_r,A_theta,A_phi

lmax = 150
nr = 280

freq =.6
eta = 0.0
E = 1e-5
Re = 0.1




'''
W,DW,Z = lib_homogeneous_prob(lmax, nr, freq, eta, E, Re)

r_i,r_o,rr = gen_radial_grid(nr, eta)
ntheta = 2*nr
theta = chebspace(0,np.pi,ntheta,inc=0)

q_r,q_theta,q_phi,kin_e = calc_kinetics(W, DW, Z, eta, 0, theta,freq)
sigma_rr,sigma_rt,sigma_rp = calc_visc_stress(W, DW, Z, eta, 0, theta, freq, E)


fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=800)


meridional_slice(np.real(q_phi),rr,theta,freq,figax=(fig,ax))

print(np.pi/(1-eta)**2*np.trapz(np.real(sigma_rp[-1])*np.real(q_phi[-1])*np.sin(theta),x=theta))

'''

'''
W_0,DW_0,Z_0,A_r,A_theta,A_phi = lib_first_order_mean(W, DW, Z, eta, E, Re)        

sigma_rr_0,sigma_rt_0,sigma_rp_0 = calc_visc_stress(W_0, DW_0, Z_0, eta, 0, theta, freq, E)
q_r_0,q_theta_0,q_phi_0,kin_e = calc_kinetics(W_0, DW_0, Z_0, eta, 0, theta,0)


fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=800)

meridional_slice(np.real(q_phi_0[:,:nr+1]), rr, theta[:nr+1], freq,figax=(fig,ax))
meridional_slice(np.real(-A_phi[:,nr:]), rr, theta[nr:], freq,figax=(fig,ax))

'''

'''
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=200)


freqs = np.linspace(0.,2,200)

lam_p = np.sqrt(np.abs(freqs+2)) * np.exp(1j*np.pi/4*np.sign(freqs+2))
lam_m = np.sqrt(np.abs(freqs-2)) * np.exp(1j*np.pi/4*np.sign(freqs-2))

K = 16/15 * (lam_p * (freqs/2+1)**2 + lam_m * (freqs/2-1)**2)
K += -32/105 * (lam_p * (freqs/2+1)**3 - lam_m * (freqs/2-1)**3)

ax.plot(freqs,np.pi/2 * E**(3/2) * Re**2 * np.real(K)/(1-eta)**2)

'''

freqs = np.linspace(0.,2,100)
powers = np.zeros(len(freqs))
for i in range(len(freqs)):
    print(i)
    W,DW,Z = lib_homogeneous_prob(lmax, nr, freqs[i], eta, E, Re)
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    ntheta = 2*nr
    theta = chebspace(0,np.pi,ntheta,inc=0)
    
    q_r,q_theta,q_phi,kin_e = calc_kinetics(W, DW, Z, eta, 0, theta,freq)
    sigma_rr,sigma_rt,sigma_rp = calc_visc_stress(W, DW, Z, eta, 0, theta, freq, E)
    
    powers[i] = np.pi/(1-eta)**2*np.trapz(np.real(sigma_rp[-1])*np.real(q_phi[-1])*np.sin(theta),x=theta)


np.savetxt('E_1_5_power_0_0.txt',powers)
np.savetxt('freqs.txt',freqs)
#ax.plot(freqs,powers)

#ax.axvline(np.sqrt(12/7))

#print(np.pi*np.trapz(np.real(sigma_rp[-1])*np.real(q_phi[-1])*np.sin(theta),x=theta))
'''
W_0,DW_0,Z_0,A_r,A_theta,A_phi = lib_first_order_mean(W, DW, Z, eta, E, Re)        

sigma_rr_0,sigma_rt_0,sigma_rp_0 = calc_visc_stress(W_0, DW_0, Z_0, eta, 0, theta, freq, E)
q_r_0,q_theta_0,q_phi_0,kin_e = calc_kinetics(W_0, DW_0, Z_0, eta, 0, theta,0)


fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=800)


#meridional_slice(np.real(q_phi_0[:,:nr+1]), rr, theta[:nr+1], freq,figax=(fig,ax))
#meridional_slice(np.real(-A_phi[:,nr:]), rr, theta[nr:], freq,figax=(fig,ax))




v_phi_mag = np.loadtxt('v_phi_fa23_lib_1_17_ave_0.txt')

theta_mag = np.loadtxt('thetas_fa23_lib_1_17.txt')
rr_mag = np.loadtxt('rr_fa23_lib_1_17.txt')
fig,ax = plt.subplots(1,1,figsize=(5,5),dpi=200)



t_0 = 90

t_idx = np.argmin(np.abs(np.radians(t_0)-theta_mag))

ax.plot(rr_mag,v_phi_mag[t_idx,:])



t_idx = np.argmin(np.abs(np.radians(t_0)-theta))


ax.plot(rr,np.real(q_phi_0[:,t_idx]))
'''


