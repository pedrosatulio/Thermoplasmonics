#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as npy
from scipy.interpolate import interp1d


# In[2]:


def Leff(a,b,isShell):
    return{
        False: 4*a/3,
        #False: a,
        #True: b-a,
        True: 4*(b**3 - a**3)/(b**2),
    }[isShell]


# In[3]:


def q_density(shell_material):
        return{
            'Ag': 5.86e28,
            'Au': 5.90e28,
        }[shell_material]


# In[4]:


def v_f(shell_material):
        return{
            'Ag': 1.39e6,
            'Au': 1.40e6,
        }[shell_material]


# In[5]:


def gamma_0(shell_material):
            return{
                'Ag': 3.22e13,
                'Au': 1.07e14,
            }[shell_material]


# In[6]:


def material_out(shell_material):
    return{
        'Ag': 'Ag_Johnson.txt',
        'Al': 'Al_McPeak.txt',
        'Au': 'Au_Johnson.txt',
        'Cu': 'Cu_Johnson.txt',
        'W': 'W_Werner.txt',
        'Si': 'Si_Schinke.txt',
        'Si_20': 'Si_Vuye_20.txt',
        'Si_100': 'Si_Vuye_100.txt',
    }[shell_material]


# In[7]:


def medium(material_medium):
    return{
        'Water': 1.33,
        'Air': 1.00,
    }[material_medium]


# In[8]:


def setupRI(shell_material,core_material,medium_material,a,b,lambda_min,lambda_max,isShell,drude,temperature):
    a /= 1e9
    b /= 1e9
    
    N = medium(medium_material)
    
    wl2 = (1e-6)*npy.loadtxt('Materials/n_'+material_out(shell_material),usecols=0)
    n2_np = npy.loadtxt('Materials/n_'+material_out(shell_material),usecols=1)
    k2_np = npy.loadtxt('Materials/k_'+material_out(shell_material),usecols=1)
    
    if isShell:
        if core_material == 'Silica':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_SiO2_Malitson.txt',usecols=0)
            n1_np = npy.loadtxt('Materials/n_SiO2_Malitson.txt',usecols=1)
            k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
        elif core_material == 'Ag':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Ag_Johnson.txt',usecols=0)
            n1_np = npy.loadtxt('Materials/n_Ag_Johnson.txt',usecols=1)
            k1_np = npy.loadtxt('Materials/k_Ag_Johnson.txt',usecols=1)
        elif core_material == 'Au':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Au_Johnson.txt',usecols=0)
            n1_np = npy.loadtxt('Materials/n_Au_Johnson.txt',usecols=1)
            k1_np = npy.loadtxt('Materials/k_Au_Johnson.txt',usecols=1)
        elif core_material == 'Si':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Si_Schinke.txt', usecols=0)
            n1_np = npy.loadtxt('Materials/n_Si_Schinke.txt', usecols=1)
            k1_np = npy.loadtxt('Materials/k_Si_Schinke.txt', usecols=1)
        elif core_material == 'Si_20':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Si_Vuye_20.txt', usecols=0)
            n1_np = npy.loadtxt('Materials/n_Si_Vuye_20.txt', usecols=1)
            k1_np = npy.loadtxt('Materials/k_Si_Vuye_20.txt', usecols=1)
        elif core_material == 'Si_100':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Si_Vuye_100.txt', usecols=0)
            n1_np = npy.loadtxt('Materials/n_Si_Vuye_100.txt', usecols=1)
            k1_np = npy.loadtxt('Materials/k_Si_Vuye_100.txt', usecols=1)
        elif core_material == 'Water':
            wl1 = wl2
            n1_np = 1.33*npy.ones(len(wl1), dtype=npy.float)
            k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
        elif core_material == 'Air':
            wl1 = wl2
            n1_np = 1*npy.ones(len(wl1), dtype=npy.float)
            k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
    else:
        wl1 = wl2
        n1_np = 0*npy.ones(len(wl1), dtype=npy.float)
        k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
    
    mu1_r = 1                                             # permeabilidade magnética relativa do núcleo (real)
    mu1_i = 0                                             # permeabilidade magnética relativa da núcleo (imaginária)
    mu2_r = 1                                             # permeabilidade magnética relativa da casca ou NP (real)
    mu2_i = 0                                             # permeabilidade magnética relativa da casca ou NP (imaginária)
    mu1 = mu1_r + mu1_i*(1.0j)                            # permeabilidade magnética complexa da nanopartícula
    mu2 = mu2_r + mu2_i*(1.0j)                            # permeabilidade magnética complexa da nanopartícula
    
    if wl2[0]<=wl1[0]:
        wl_min = wl1[0]
    else:
        wl_min = wl2[0]
    if wl2[len(wl2)-1]<=wl1[len(wl1)-1]:
        wl_max = wl2[len(wl2)-1]
    else:
        wl_max = wl1[len(wl1)-1]
    
    wl = npy.linspace(wl_min, wl_max, num=600, endpoint=True)
    
    wl = npy.delete(wl,len(wl)-1)
    fn1_np = interp1d(wl1, n1_np, kind='cubic')
    fk1_np = interp1d(wl1, k1_np, kind='cubic')
    fn2_np = interp1d(wl2, n2_np, kind='cubic')
    fk2_np = interp1d(wl2, k2_np, kind='cubic')
    
    if (core_material == 'Si' and temperature != 0):
        N1_bulk = (fn1_np(wl) + fk1_np(wl)*(1.0j))*npy.sqrt(mu1)
        eps1_bulk = N1_bulk**2
        eps0 = 8.8541878128e-12
        q_e = 1.60217662e-19
        kB = 1.38064852e-23
        Eg0 = 1.166
        alpha = 4.73e-4
        beta = 636
        Nc300 = 2.82e25
        Nv300 = 1.83e25
        mue300 = 0.143
        muh300 = 0.046
        omega = 2*math.pi*(3e8)/wl
        Eg = Eg0 - (alpha*(temperature**2))/(beta + temperature)
        Nc = Nc300*((temperature/300)**(3/2))
        Nv = Nv300*((temperature/300)**(3/2))
        Ni = npy.sqrt(Nc*Nv)*npy.exp(-Eg/(2*kB*temperature))
        mue = mue300*((temperature/300)**(-2))
        muh = muh300*((temperature/300)**(-2.18))
        sigma300 = npy.sqrt(Nc300*Nv300)*npy.exp(-Eg/(2*kB*300))*q_e*(mue300 + muh300)
        sigma = Ni*q_e*(mue + muh)
        eps1 = npy.zeros(len(wl), dtype=npy.cfloat)
        N1 = npy.zeros(len(wl), dtype=npy.cfloat)
        for h in range(0,len(wl)-1,1):
            eps1[h] = eps1_bulk[h]**2 - (1.0j)*(4*math.pi*sigma300)/(omega[h]*eps0) + (1.0j)*(4*math.pi*sigma)/(omega[h]*eps0)
            #eps1[h] = eps1_bulk[h]**2 - (1.0j)*(4*math.pi*sigma300)/(omega[h]) + (1.0j)*(4*math.pi*sigma)/(omega[h])
            N1[h] = npy.sqrt(eps1[h])
    else:
        N1 = (fn1_np(wl) + fk1_np(wl)*(1.0j))*npy.sqrt(mu1)
    
    if (shell_material == 'Si' and temperature != 0):
        N2_bulk = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)
        eps2_bulk = N2_bulk**2
        eps0 = 8.8541878128e-12
        q_e = 1.60217662e-19
        kB = 1.38064852e-23
        Eg0 = 1.166
        alpha = 4.73e-4
        beta = 636
        Nc300 = 2.82e25
        Nv300 = 1.83e25
        mue300 = 0.143
        muh300 = 0.046
        omega = 2*math.pi*(3e8)/wl
        Eg = Eg0 - (alpha*(temperature**2))/(beta + temperature)
        Nc = Nc300*((temperature/300)**(3/2))
        Nv = Nv300*((temperature/300)**(3/2))
        Ni = npy.sqrt(Nc*Nv)*npy.exp(-Eg/(2*kB*temperature))
        mue = mue300*((temperature/300)**(-2))
        muh = muh300*((temperature/300)**(-2.18))
        sigma300 = npy.sqrt(Nc300*Nv300)*npy.exp(-Eg/(2*kB*300))*q_e*(mue300 + muh300)
        sigma = Ni*q_e*(mue + muh)
        eps2 = npy.zeros(len(wl), dtype=npy.cfloat)
        N2 = npy.zeros(len(wl), dtype=npy.cfloat)
        for h in range(0,len(wl)-1,1):
            eps2[h] = eps2_bulk[h] - (1.0j)*(4*math.pi*sigma300)/(omega[h]*eps0) + (1.0j)*(4*math.pi*sigma)/(omega[h]*eps0)
            #eps2[h] = eps2_bulk[h] - (1.0j)*(4*math.pi*sigma300)/(omega[h]) + (1.0j)*(4*math.pi*sigma)/(omega[h])
            N2[h] = npy.sqrt(eps2[h])
    else:
        if drude:
            N2_pre = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)
        else:
            N2 = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)
        
        if drude:
            eps0 = 8.8541878128e-12
            q_e = -1.60217662e-19
            m_e = 9.10938356e-31
            omega_p = npy.sqrt(q_density(shell_material)*q_e*q_e/(eps0*m_e))
            A = 1
            omega = 2*math.pi*(3e8)/wl
            eps_intra = npy.zeros(len(wl), dtype=npy.cfloat)
            eps_corr = npy.zeros(len(wl), dtype=npy.cfloat)
            N2 = npy.zeros(len(wl), dtype=npy.cfloat)
            for h in range(0,len(wl)-1,1):
                eps_intra[h] = omega_p*omega_p/(omega[h]*omega[h] + (1.0j)*omega[h]*gamma_0(shell_material))
                eps_corr[h] = omega_p*omega_p/(omega[h]*omega[h] + (1.0j)*omega[h]*(gamma_0(shell_material)+A*v_f(shell_material)/Leff(a,b,isShell)))
                N2[h] = npy.sqrt(N2_pre[h]*N2_pre[h] + eps_intra[h] - eps_corr[h])
    
    return wl, N, N1, N2, mu1, mu2


# In[ ]:


def setupRIu(shell_material,core_material,medium_material,a,b,wl,isShell,drude,temperature):
    a /= 1e9
    b /= 1e9
    wl /= 1e9
    
    N = medium(medium_material)
    
    wl2 = (1e-6)*npy.loadtxt('Materials/n_'+material_out(shell_material),usecols=0)
    n2_np = npy.loadtxt('Materials/n_'+material_out(shell_material),usecols=1)
    k2_np = npy.loadtxt('Materials/k_'+material_out(shell_material),usecols=1)
    
    if isShell:
        if core_material == 'Silica':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_SiO2_Malitson.txt',usecols=0)
            n1_np = npy.loadtxt('Materials/n_SiO2_Malitson.txt',usecols=1)
            k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
        elif core_material == 'Ag':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Ag_Johnson.txt',usecols=0)
            n1_np = npy.loadtxt('Materials/n_Ag_Johnson.txt',usecols=1)
            k1_np = npy.loadtxt('Materials/k_Ag_Johnson.txt',usecols=1)
        elif core_material == 'Au':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Au_Johnson.txt',usecols=0)
            n1_np = npy.loadtxt('Materials/n_Au_Johnson.txt',usecols=1)
            k1_np = npy.loadtxt('Materials/k_Au_Johnson.txt',usecols=1)
        elif core_material == 'Water':
            wl1 = wl2
            n1_np = 1.33*npy.ones(len(wl1), dtype=npy.float)
            k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
        elif core_material == 'Air':
            wl1 = wl2
            n1_np = 1*npy.ones(len(wl1), dtype=npy.float)
            k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
    else:
        wl1 = wl2
        n1_np = 0*npy.ones(len(wl1), dtype=npy.float)
        k1_np = 0*npy.ones(len(wl1), dtype=npy.float)
    
    mu1_r = 1                                             # permeabilidade magnética relativa do núcleo (real)
    mu1_i = 0                                             # permeabilidade magnética relativa da núcleo (imaginária)
    mu2_r = 1                                             # permeabilidade magnética relativa da casca ou NP (real)
    mu2_i = 0                                             # permeabilidade magnética relativa da casca ou NP (imaginária)
    mu1 = mu1_r + mu1_i*(1.0j)                            # permeabilidade magnética complexa da nanopartícula
    mu2 = mu2_r + mu2_i*(1.0j)                            # permeabilidade magnética complexa da nanopartícula
    
    fn1_np = interp1d(wl1, n1_np, kind='cubic')
    fk1_np = interp1d(wl1, k1_np, kind='cubic')
    fn2_np = interp1d(wl2, n2_np, kind='cubic')
    fk2_np = interp1d(wl2, k2_np, kind='cubic')
    N1 = (fn1_np(wl) + fk1_np(wl)*(1.0j))*npy.sqrt(mu1)
    
    if (core_material == 'Si'):
        N1_bulk = (fn1_np(wl) + fk1_np(wl)*(1.0j))*npy.sqrt(mu1)
        eps1_bulk = N1_bulk**2
        eps0 = 8.8541878128e-12
        q_e = 1.60217662e-19
        kB = 1.38064852e-23
        Eg0 = 1.166
        alpha = 4.73e-4
        beta = 636
        Nc300 = 2.82e25
        Nv300 = 1.83e25
        mue300 = 0.143
        muh300 = 0.046
        omega = 2*math.pi*(3e8)/wl
        Eg = Eg0 - (alpha*(temperature**2))/(beta + temperature)
        Nc = Nc300*((temperature/300)**(3/2))
        Nv = Nv300*((temperature/300)**(3/2))
        Ni = npy.sqrt(Nc*Nv)*npy.exp(-Eg/(2*kB*temperature))
        mue = mue300*((temperature/300)**(-2))
        muh = muh300*((temperature/300)**(-2.18))
        sigma300 = npy.sqrt(Nc300*Nv300)*npy.exp(-Eg/(2*kB*300))*q_e*(mue300 + muh300)
        sigma = Ni*q_e*(mue + muh)
        eps1 = npy.zeros(len(wl), dtype=npy.cfloat)
        N1 = npy.zeros(len(wl), dtype=npy.cfloat)
        for h in range(0,len(wl)-1,1):
            eps1[h] = eps1_bulk[h] - (1.0j)*(4*math.pi*sigma300)/(omega[h]*eps0) + (1.0j)*(4*math.pi*sigma)/(omega[h]*eps0)
            #eps1[h] = eps1_bulk[h] - (1.0j)*(4*math.pi*sigma300)/(omega[h]) + (1.0j)*(4*math.pi*sigma)/(omega[h])
            N1[h] = npy.sqrt(eps1[h])
    else:
        N1 = (fn1_np(wl) + fk1_np(wl)*(1.0j))*npy.sqrt(mu1)
    
    if (shell_material == 'Si'):
        N2_bulk = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)
        eps2_bulk = N2_bulk**2
        eps0 = 8.8541878128e-12
        q_e = 1.60217662e-19
        kB = 1.38064852e-23
        Eg0 = 1.166
        alpha = 4.73e-4
        beta = 636
        Nc300 = 2.82e25
        Nv300 = 1.83e25
        mue300 = 0.143
        muh300 = 0.046
        omega = 2*math.pi*(3e8)/wl
        Eg = Eg0 - (alpha*(temperature**2))/(beta + temperature)
        Nc = Nc300*((temperature/300)**(3/2))
        Nv = Nv300*((temperature/300)**(3/2))
        Ni = npy.sqrt(Nc*Nv)*npy.exp(-Eg/(2*kB*temperature))
        mue = mue300*((temperature/300)**(-2))
        muh = muh300*((temperature/300)**(-2.18))
        sigma300 = npy.sqrt(Nc300*Nv300)*npy.exp(-Eg/(2*kB*300))*q_e*(mue300 + muh300)
        sigma = Ni*q_e*(mue + muh)
        eps2 = npy.zeros(len(wl), dtype=npy.cfloat)
        N2 = npy.zeros(len(wl), dtype=npy.cfloat)
        for h in range(0,len(wl)-1,1):
            eps2[h] = eps2_bulk[h] - (1.0j)*(4*math.pi*sigma300)/(omega[h]*eps0) + (1.0j)*(4*math.pi*sigma)/(omega[h]*eps0)
            #eps2[h] = eps2_bulk[h] - (1.0j)*(4*math.pi*sigma300)/(omega[h]) + (1.0j)*(4*math.pi*sigma)/(omega[h])
            N2[h] = npy.sqrt(eps2[h])
    else:
        if drude:
            N2_pre = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)
        else:
            N2 = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)

        if drude:
            eps0 = 8.8541878128e-12
            q_e = -1.60217662e-19
            m_e = 9.10938356e-31
            omega_p = npy.sqrt(q_density(shell_material)*q_e*q_e/(eps0*m_e))
            A = 1
            omega = 2*math.pi*(3e8)/wl
            eps_intra = omega_p*omega_p/(omega*omega + (1.0j)*omega*gamma_0(shell_material))
            eps_corr = omega_p*omega_p/(omega*omega + (1.0j)*omega*(gamma_0(shell_material)+A*v_f(shell_material)/Leff(a,b,isShell)))
            N2 = npy.sqrt(N2_pre*N2_pre + eps_intra - eps_corr)
        
    return N, N1, N2, mu1, mu2

