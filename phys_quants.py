import numpy as np
from matrix_solver import gen_radial_grid
from special_funcs import sphrharm,chebspace
from finite_differences import df1_lil 

def calc_kinetics(W,DW,Z,eta,m,theta,freq,vectorized= 1):
    
    lmax,nr = np.shape(W)
    
    if vectorized:
        ntheta = len(theta)
        
        r_i,r_o,rr = gen_radial_grid(nr, eta)
        orr = 1/rr
        
        kin_e = np.zeros((nr,ntheta),dtype='float')
    
        q_r = np.zeros((nr,ntheta),dtype='complex')
        q_theta = np.zeros_like(q_r)
        q_phi = np.zeros_like(q_r)
    
        YY = np.zeros((lmax+1,ntheta),dtype=complex)
    
        for l in range(m,lmax+1):
            
            YY[l,:] = sphrharm(l,m,theta,0)
        
        ll = np.linspace(0,lmax,lmax+1)
        
        
        c = np.zeros_like(ll)
        for l in range(m,lmax+1):
            c[l] = np.sqrt((l-m+1)*(l+m+1)/(2*l+1)/(2*l+3))
    
    
        for l in range(1,lmax):
                
            #print(l*(l+1)/0.65**2 * sphrharm(l,0,np.pi/2,0),np.real(W[l,-1]))
            
            
            q_r += YY[l][np.newaxis,:] * W[l][:,np.newaxis] * ll[l] * (ll[l]+1) * orr[:,np.newaxis]**2
            
            q_theta += orr[:,np.newaxis] * DW[l][:,np.newaxis] /np.sin(theta[np.newaxis,:]) * (ll[l] * c[l] * YY[l+1][np.newaxis,:]-(ll[l]+1) * c[l-1] * YY[l-1][np.newaxis,:])
            q_theta += 1j*m * orr[:,np.newaxis] / np.sin(theta[np.newaxis,:]) * YY[l] * Z[l][:,np.newaxis]
    
            q_phi += -Z[l][:,np.newaxis] * orr[:,np.newaxis] / np.sin(theta[np.newaxis,:]) * (ll[l] * c[l] * YY[l+1][np.newaxis,:]-(ll[l]+1) * c[l-1] * YY[l-1][np.newaxis,:])
            q_phi += 1j*m * orr[:,np.newaxis] / np.sin(theta[np.newaxis,:]) * YY[l] * DW[l][:,np.newaxis]
    
        
        kin_e = 0.5* (q_r**2 + q_theta**2+q_phi**2)
        
        return q_r,q_theta,q_phi,kin_e


def calc_kin_energy(W,DW,Z,eta,m,theta,freq):
    
    q_r,q_theta,q_phi,kin_den = calc_kinetics(W, DW, Z, eta, m, theta, freq)
    
    lmax,nr = np.shape(W)
    ntheta = len(theta)
    nphi  = 2* ntheta
    
    
    if m == 0:
        
        r_i,r_o,rr = gen_radial_grid(nr, eta)
        
        qs_re = np.real(q_r)**2+np.real(q_theta)**2+np.real(q_phi)**2
        qs_im = np.imag(q_r)**2+np.imag(q_theta)**2+np.imag(q_phi)**2
        
        qs_mix = np.real(q_r) * np.imag(q_r) + np.real(q_theta) * np.imag(q_theta) + np.real(q_phi) * np.imag(q_phi)
        
        Q_re = np.pi * np.trapz(np.trapz(qs_re*rr[:,np.newaxis]**2,x = rr,axis=0)*np.sin(theta[:]),x=theta,axis=0)
        Q_im = np.pi * np.trapz(np.trapz(qs_im*rr[:,np.newaxis]**2,x = rr,axis=0)*np.sin(theta[:]),x=theta,axis=0)
        Q_mix = np.pi * np.trapz(np.trapz(qs_mix*rr[:,np.newaxis]**2,x = rr,axis=0)*np.sin(theta[:]),x=theta,axis=0)
        
        return Q_re,Q_im,Q_mix
    
    else:
        
        phi = np.linspace(0,2*np.pi,nphi)
        
        r_i,r_o,rr = gen_radial_grid(nr, eta)
        
        q_r = q_r[:,:,np.newaxis] * np.exp(1j*m*phi[np.newaxis,np.newaxis,:])
        q_theta = q_theta[:,:,np.newaxis] * np.exp(1j*m*phi[np.newaxis,np.newaxis,:])
        q_phi = q_phi[:,:,np.newaxis] * np.exp(1j*m*phi[np.newaxis,np.newaxis,:])
        
        qs_re = np.real(q_r)**2+np.real(q_theta)**2+np.real(q_phi)**2
        qs_im = np.imag(q_r)**2+np.imag(q_theta)**2+np.imag(q_phi)**2
        
        qs_mix = np.real(q_r) * np.imag(q_r) + np.real(q_theta) * np.imag(q_theta) + np.real(q_phi) * np.imag(q_phi)
        
        
        Q_re = 0.5 * np.trapz(np.trapz(np.trapz(qs_re*rr[:,np.newaxis,np.newaxis]**2,x = rr,axis=0)*np.sin(theta[:,np.newaxis]),x=theta,axis=0),x=phi)
        Q_im = 0.5 * np.trapz(np.trapz(np.trapz(qs_im*rr[:,np.newaxis,np.newaxis]**2,x = rr,axis=0)*np.sin(theta[:,np.newaxis]),x=theta,axis=0),x=phi)
        Q_mix = 0.5 * np.trapz(np.trapz(np.trapz(qs_mix*rr[:,np.newaxis,np.newaxis]**2,x = rr,axis=0)*np.sin(theta[:,np.newaxis]),x=theta,axis=0),x=phi)
        
        return Q_re,Q_im,Q_mix

def calc_visc_stress(W,DW,Z,eta,m,theta,freq,E,t=0):
    
    lmax,nr = np.shape(W)
    ntheta = len(theta)
    
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    orr = 1/rr
    
    sigma_rr = np.zeros((nr,ntheta),dtype=complex)
    sigma_rt = np.zeros_like(sigma_rr)
    sigma_rp = np.zeros_like(sigma_rr)
    
    YY = np.zeros((lmax+1,ntheta),dtype=complex)
    
    DDW = np.zeros_like(W)
    DZ = np.zeros_like(W)
    df1 = df1_lil(rr)
    
    for l in range(lmax):
        
        DDW[l,:] = df1.dot(DW[l,:])
        DZ[l,:] = df1.dot(Z[l,:])
        
    for l in range(m,lmax+1):
        
        YY[l,:] = sphrharm(l,m,theta,0)
    
    ll = np.linspace(0,lmax,lmax+1)
    c = np.zeros_like(ll)
    for l in range(m,lmax+1):
        c[l] = np.sqrt((l-m+1)*(l+m+1)/(2*l+1)/(2*l+3))
        
    for l in range(lmax):
        
        sigma_rr += 2 * E * ll[l] * (ll[l]+1) * orr[:,np.newaxis]**2 * DW[l][:,np.newaxis]
        
        sigma_rt += E * orr[:,np.newaxis]/ np.sin(theta[np.newaxis,:])  * (ll[l] * c[l] * YY[l+1][np.newaxis,:]-(ll[l]+1) * c[l-1] * YY[l-1][np.newaxis,:]) * ( ll[l] * (ll[l]+1) * orr[:,np.newaxis]**2 * W[l][:,np.newaxis] + DDW[l][:,np.newaxis] - 2 * orr[:,np.newaxis] * DW[l][:,np.newaxis])
        sigma_rt += 1j * m * E * orr[:,np.newaxis] / np.sin(theta[np.newaxis,:]) * YY[l][np.newaxis,:] * (DZ[l][:,np.newaxis]-2*orr[:,np.newaxis] * Z[l][:,np.newaxis])
    
        sigma_rp += 1j * m * E * orr[:,np.newaxis] / np.sin(theta[np.newaxis,:]) * YY[l][np.newaxis,:] * ( ll[l] * (ll[l]+1) * orr[:,np.newaxis]**2 * W[l][:,np.newaxis] + DDW[l][:,np.newaxis] - 2 * orr[:,np.newaxis] * DW[l][:,np.newaxis])
        sigma_rp += -E * orr[:,np.newaxis]/ np.sin(theta[np.newaxis,:]) * (ll[l] * c[l] * YY[l+1][np.newaxis,:]-(ll[l]+1) * c[l-1] * YY[l-1][np.newaxis,:]) * (DZ[l][:,np.newaxis]-2*orr[:,np.newaxis] * Z[l][:,np.newaxis])
    
    return sigma_rr,sigma_rt,sigma_rp


def calc_advection(W,DW,Z,eta,m,theta,freq,E):
    
    im = np.imag
    re = np.real
    
    # for now only m == 0
    q_r,q_theta,q_phi,kin_den = calc_kinetics(W, DW, Z, eta, m, theta, freq)
    
    lmax,nr = np.shape(W)
    ntheta = len(theta)
    r_i,r_o,rr = gen_radial_grid(nr, eta)
    orr = 1/rr
    
    df1_theta = df1_lil(theta,cheb=0)
    df1_r = df1_lil(rr)
    
    q_r_dtheta = np.zeros_like(q_r)
    q_r_dr = np.zeros_like(q_r)
    
    q_theta_dtheta = np.zeros_like(q_r)
    q_theta_dr = np.zeros_like(q_r)
    
    q_phi_dtheta = np.zeros_like(q_r)
    q_phi_dr = np.zeros_like(q_r)
    
    
    for i in range(nr):
        
        q_r_dtheta[i,:] = df1_theta.dot(q_r[i,:])
        q_theta_dtheta[i,:] = df1_theta.dot(q_theta[i,:])
        q_phi_dtheta[i,:] = df1_theta.dot(q_phi[i,:])
        
    for i in range(ntheta):
        
        q_r_dr[:,i] = df1_r.dot(q_r[:,i])
        q_theta_dr[:,i] = df1_r.dot(q_theta[:,i])
        q_phi_dr[:,i] = df1_r.dot(q_phi[:,i])
    
    adv_phi_re = re(q_r) * re(q_phi_dr) + re(q_theta) * orr[:,np.newaxis] * re(q_phi_dtheta)
    adv_phi_re += re(q_phi) * orr[:,np.newaxis] * (re(q_r)+re(q_theta) * np.cos(theta[np.newaxis,:])/np.sin(theta[np.newaxis,:]))
    
    adv_phi_im = im(q_r) * im(q_phi_dr) + im(q_theta) * orr[:,np.newaxis] * im(q_phi_dtheta)
    adv_phi_im += im(q_phi) * orr[:,np.newaxis] * (im(q_r)+im(q_theta) * np.cos(theta[np.newaxis,:])/np.sin(theta[np.newaxis,:]))
    
    adv_r_re = re(q_r) * re(q_r_dr) + re(q_theta) * orr[:,np.newaxis]*(re(q_r_dtheta) - re(q_theta)) - re(q_phi)**2 * orr[:,np.newaxis]
    adv_r_im = im(q_r) * im(q_r_dr) + im(q_theta) * orr[:,np.newaxis]*(im(q_r_dtheta) - im(q_theta)) - im(q_phi)**2 * orr[:,np.newaxis]
    
    adv_theta_re = re(q_r) * re(q_theta_dr) + re(q_theta) * orr[:,np.newaxis] * ( re(q_theta_dtheta) + re(q_r)) - re(q_phi)**2 * orr[:,np.newaxis] * np.cos(theta[np.newaxis,:]) / np.sin(theta[np.newaxis,:]) 
    adv_theta_im = im(q_r) * im(q_theta_dr) + im(q_theta) * orr[:,np.newaxis] * ( im(q_theta_dtheta) + im(q_r)) - im(q_phi)**2 * orr[:,np.newaxis] * np.cos(theta[np.newaxis,:]) / np.sin(theta[np.newaxis,:]) 
    
    
    return adv_r_re,adv_r_im,adv_theta_re,adv_theta_im,adv_phi_re,adv_phi_im

    



