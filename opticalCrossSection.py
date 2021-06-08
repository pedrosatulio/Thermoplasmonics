#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as npy
from scipy import special as sp


# In[2]:


def sphereCoeffs(wl,N2,N,mu1,a):
    wl /= 1e9
    a /= 1e9
    k = (2*math.pi*N)/wl                                    # número de onda do meio
    m = N2/N                                                # índice de refração relativo da casca ou NP
    x = k*a                                                 # parâmetro de tamanho (esfera)
    v_max = math.ceil(x + 4.05*x**(1/3) + 2)                # ordem máxima necessária (esfera)
    a_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de espalhamento a
    b_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de espalhamento b
    c_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo interno c
    d_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo interno d
    # Computação das constantes de espalhamento e das constantes de campo interno 
    for v in range(1,v_max,1):
        # Cálculo dos coeficientes do sistema linear
        p12 = sp.spherical_jn(v,x)+(1.0j)*sp.spherical_yn(v,x)
        p13 = sp.spherical_jn(v,m*x)
        p22 = mu1*(x*(sp.spherical_jn(v-1,x)+(1.0j)*sp.spherical_yn(v-1,x))-v*(sp.spherical_jn(v,x)+(1.0j)*sp.spherical_yn(v,x)))
        p23 = m*x*sp.spherical_jn(v-1,m*x)-v*sp.spherical_jn(v,m*x)
        p31 = mu1*(sp.spherical_jn(v,x)+(1.0j)*sp.spherical_yn(v,x))
        p34 = m*sp.spherical_jn(v,m*x)
        p41 = m*(x*(sp.spherical_jn(v-1,x)+(1.0j)*sp.spherical_yn(v-1,x))-v*(sp.spherical_jn(v,x)+(1.0j)*sp.spherical_yn(v,x)))
        p44 = m*x*sp.spherical_jn(v-1,m*x)-v*sp.spherical_jn(v,m*x)
        q1 = sp.spherical_jn(v,x)
        q2 = mu1*(x*sp.spherical_jn(v-1,x)-v*sp.spherical_jn(v,x))
        q3 = mu1*sp.spherical_jn(v,x)
        q4 = m*(x*sp.spherical_jn(v-1,x)-v*sp.spherical_jn(v,x))
        # Solução do sistema linear
        #               an   bn   cn   dn
        A = npy.array([[0,   p12, p13, 0  ],
                       [0,   p22, p23, 0  ],
                       [p31, 0,   0,   p34],
                       [p41, 0,   0,   p44]])
        B = npy.array([q1,q2,q3,q4])
        X = npy.linalg.solve(A, B)
        a_n[v] = X[0]
        b_n[v] = X[1]
        c_n[v] = X[2]
        d_n[v] = X[3]
    return a_n, b_n, c_n, d_n


# In[3]:


def shellCoeffs(wl,N1,N2,N,mu1,mu2,a,b):
    wl /= 1e9
    a /= 1e9
    b /= 1e9
    k = (2*math.pi*N)/wl                                    # número de onda do meio
    m1 = N1/N                                               # índice de refração relativo do núcleo
    m2 = N2/N                                               # índice de refração relativo da casca ou NP
    x = k*a                                                 # parâmetro de tamanho (esfera)
    y = k*b                                                 # parâmetro de tamanho (casca)
    v_max = math.ceil(y + 4.05*y**(1/3) + 2)                # ordem máxima necessária (casca)
    # Parâmetros fronteira núcleo-casca
    psi_m1x = npy.empty(v_max, dtype=npy.cfloat)            # array para Riccati-Bessel do 1º tipo no núcleo
    psi_m2x = npy.empty(v_max, dtype=npy.cfloat)            # array para Riccati-Bessel do 1º tipo na casca
    chi_m2x = npy.empty(v_max, dtype=npy.cfloat)            # array para Riccati-Bessel do 2º tipo na casca
    dpsi_m1x = npy.empty(v_max, dtype=npy.cfloat)           # array derivada para Riccati-Bessel do 1º tipo no núcleo
    dpsi_m2x = npy.empty(v_max, dtype=npy.cfloat)           # array derivada para Riccati-Bessel do 1º tipo na casca
    dchi_m2x = npy.empty(v_max, dtype=npy.cfloat)           # array derivada para Riccati-Bessel do 2º tipo na casca
    # Parâmetros fronteira casca-meio
    csi_y = npy.empty(v_max, dtype=npy.cfloat)              # array para Riccati-Bessel do 1º tipo (em hankel esférico do 1º tipo) no meio
    psi_y = npy.empty(v_max, dtype=npy.cfloat)              # array para Riccati-Bessel do 1º tipo no meio
    psi_m2y = npy.empty(v_max, dtype=npy.cfloat)            # array para Riccati-Bessel do 1º tipo na casca
    chi_m2y = npy.empty(v_max, dtype=npy.cfloat)            # array para Riccati-Bessel do 2º tipo na casca
    dcsi_y = npy.empty(v_max, dtype=npy.cfloat)             # array derivada para Riccati-Bessel do 1º tipo (em hankel esférico do 1º tipo) no meio
    dcsi_y = npy.empty(v_max, dtype=npy.cfloat)             # array derivada para Riccati-Bessel do 1º tipo no meio
    dpsi_y = npy.empty(v_max, dtype=npy.cfloat)             # array derivada para Riccati-Bessel do 1º tipo na casca
    dpsi_m2y = npy.empty(v_max, dtype=npy.cfloat)           # array derivada para Riccati-Bessel do 1º tipo na casca
    dchi_m2y = npy.empty(v_max, dtype=npy.cfloat)           # array derivada para Riccati-Bessel do 2º tipo na casca
    # Constantes de espalhamento e constantes de campo
    a_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de espalhamento a
    b_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de espalhamento b
    c_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo c
    d_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo d
    f_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo f
    g_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo g
    v_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo v
    w_n = npy.empty(v_max, dtype=npy.cfloat)                # array coeficiente de campo w
    # Computação das constantes de espalhamento e das constantes de campo
    for v in range(0,v_max,1) :
        psi_m1x[v] = m1*x*sp.spherical_jn(v,m1*x)
        psi_m2x[v] = m2*x*sp.spherical_jn(v,m2*x)
        chi_m2x[v] = -m2*x*sp.spherical_yn(v,m2*x)
        csi_y[v] = y*sp.spherical_jn(v,y) + y*sp.spherical_yn(v,y)*(1.0j)
        psi_y[v] = y*sp.spherical_jn(v,y)
        psi_m2y[v] = m2*y*sp.spherical_jn(v,m2*y)
        chi_m2y[v] = -m2*y*sp.spherical_yn(v,m2*y)
    for v in range(0,v_max,1) :
        if v == 0 :
            dpsi_m1x[v] = sp.spherical_jn(0,m1*x) - psi_m1x[1]
            dpsi_m2x[v] = sp.spherical_jn(0,m2*x) - psi_m2x[1]
            dchi_m2x[v] = -sp.spherical_yn(0,m2*x) - chi_m2x[1]
            dcsi_y[v] = sp.spherical_jn(0,y) + sp.spherical_yn(0,y)*(1.0j) - csi_y[1]
            dpsi_y[v] = sp.spherical_jn(0,y) - psi_y[1]
            dpsi_m2y[v] = sp.spherical_jn(0,m2*y) - psi_m2y[1]
            dchi_m2y[v] = -sp.spherical_yn(0,m2*y) - chi_m2y[1]
        else :
            dpsi_m1x[v] = psi_m1x[v-1] - (v/(m1*x))*psi_m1x[v]
            dpsi_m2x[v] = psi_m2x[v-1] - (v/(m2*x))*psi_m2x[v]
            #dchi_m2x[v] = chi_m2x[v-1] - (v/(m2*x))*chi_m2x[v]
            dchi_m2x[v] = v*sp.spherical_yn(v,m2*x) + chi_m2x[v-1]
            dcsi_y[v] = csi_y[v-1] - (v/y)*csi_y[v]
            dpsi_y[v] = psi_y[v-1] - (v/y)*psi_y[v]
            dpsi_m2y[v] = psi_m2y[v-1] - (v/(m2*y))*psi_m2y[v]
            #dchi_m2y[v] = chi_m2y[v-1] - (v/(m2*y))*chi_m2y[v]
            dchi_m2y[v] = v*sp.spherical_yn(v,m2*y) + chi_m2y[v-1]
    for v in range(1,v_max,1) :
        # Cálculo dos coeficientes do sistema linear
        p13 = -m2*psi_m1x[v]
        p15 = m1*psi_m2x[v]
        p17 = -m1*chi_m2x[v]
        p24 = m2*dpsi_m1x[v]
        p26 = -m1*dpsi_m2x[v]
        p28 = m1*dchi_m2x[v]
        p33 = mu2*dpsi_m1x[v]
        p35 = -mu1*dpsi_m2x[v]
        p37 = mu1*dchi_m2x[v]
        p44 = -mu2*psi_m1x[v]
        p46 = mu1*psi_m2x[v]
        p48 = -mu1*chi_m2x[v]
        p51 = -m2*dcsi_y[v]
        p56 = -dpsi_m2y[v]
        p58 = dchi_m2y[v]
        p62 = m2*csi_y[v]
        p65 = psi_m2y[v]
        p67 = -chi_m2y[v]
        p71 = -mu2*csi_y[v]
        p76 = -psi_m2y[v]
        p78 = chi_m2y[v]
        p82 = mu2*dcsi_y[v]
        p85 = dpsi_m2y[v]
        p87 = -dchi_m2y[v]
        q5 = -m2*dpsi_y[v]
        q6 = m2*psi_y[v]
        q7 = -mu2*psi_y[v]
        q8 = mu2*dpsi_y[v]
        # Solução do sistema linear
        #               an   bn   cn   dn   fn   gn   vn   wn
        A = npy.array([[0,   0,   p13, 0,   p15, 0,   p17, 0  ],
                       [0,   0,   0,   p24, 0,   p26, 0,   p28],
                       [0,   0,   p33, 0,   p35, 0,   p37, 0  ],
                       [0,   0,   0,   p44, 0,   p46, 0,   p48],
                       [p51, 0,   0,   0,   0,   p56, 0,   p58],
                       [0,   p62, 0,   0,   p65, 0,   p67, 0  ],
                       [p71, 0,   0,   0,   0,   p76, 0,   p78],
                       [0,   p82, 0,   0,   p85, 0,   p87, 0  ]])
        B = npy.array([0,0,0,0,q5,q6,q7,q8])
        X = npy.linalg.solve(A, B)
        a_n[v] = X[0]
        b_n[v] = X[1]
        c_n[v] = X[2]
        d_n[v] = X[3]
        f_n[v] = X[4]
        g_n[v] = X[5]
        v_n[v] = X[6]
        w_n[v] = X[7]
    return a_n, b_n, c_n, d_n, f_n, g_n, v_n, w_n


# In[4]:


def sphereCSu(wl,N2,N,mu1,a):
    # Parâmetros gerais
    a_n, b_n, c_n, d_n = sphereCoeffs(wl,N2,N,mu1,a)
    wl /= 1e9
    a /= 1e9
    k = (2*math.pi*N)/wl                                    # número de onda do meio
    x = k*a                                                 # parâmetro de tamanho (esfera)
    v_max = math.ceil(x + 4.05*x**(1/3) + 2)                # ordem máxima necessária (esfera)
    # Computação das seções ópticas
    sca_p = 0
    ext_p = 0
    bck_p = 0
    for v in range(1,v_max,1) :
        sca_p += (2*v + 1)*(abs(a_n[v])**2 + abs(b_n[v])**2)
        ext_p += (2*v + 1)*(a_n[v].real + b_n[v].real)
        bck_p += abs((2*v + 1)*((-1)**v)*(a_n[v] - b_n[v]))**2
    Csca = (2*math.pi/(k*k))*sca_p
    Cext = (2*math.pi/(k*k))*ext_p
    Cbck = (math.pi/(k*k))*bck_p
    Cabs = Cext - Csca    
    return Csca, Cext, Cabs, Cbck


# In[5]:


def shellCSu(wl,N1,N2,N,mu1,mu2,a,b):
    # Parâmetros gerais
    a_n, b_n, c_n, d_n, f_n, g_n, v_n, w_n = shellCoeffs(wl,N1,N2,N,mu1,mu2,a,b)
    wl /= 1e9
    a /= 1e9
    b /= 1e9
    k = (2*math.pi*N)/wl                                    # número de onda do meio
    m1 = N1/N                                               # índice de refração relativo do núcleo
    m2 = N2/N                                               # índice de refração relativo da casca ou NP
    x = k*a                                                 # parâmetro de tamanho (esfera)
    y = k*b                                                 # parâmetro de tamanho (casca)
    v_max = math.ceil(y + 4.05*y**(1/3) + 2)                # ordem máxima necessária (casca)
    # Computação das seções ópticas
    sca_p = 0
    ext_p = 0
    bck_p = 0
    for v in range(1,v_max,1) :
        sca_p += (2*v + 1)*(abs(a_n[v])**2 + abs(b_n[v])**2)
        ext_p += (2*v + 1)*(a_n[v].real + b_n[v].real)
        bck_p += abs((2*v + 1)*((-1)**v)*(a_n[v] - b_n[v]))**2
    Csca = (2*math.pi/(k*k))*sca_p
    Cext = (2*math.pi/(k*k))*ext_p
    Cbck = (math.pi/(k*k))*bck_p
    Cabs = Cext - Csca    
    return Csca, Cext, Cabs, Cbck


# In[6]:


def sphereCS(wl,N2,N,mu1,a):
    wl /= 1e9
    Csca = npy.empty(len(wl), dtype=npy.float)
    Cext = npy.empty(len(wl), dtype=npy.float)
    Cbck = npy.empty(len(wl), dtype=npy.float)
    Cabs = npy.empty(len(wl), dtype=npy.float)
    for index in range(0,len(wl)-1,1):
        temp = sphereCSu(wl[index]*(1e9),N2[index],N,mu1,a)
        Csca[index] = temp[0]*(1e18)
        Cext[index] = temp[1]*(1e18)
        Cabs[index] = temp[2]*(1e18)
        Cbck[index] = temp[3]*(1e18)
    return Csca, Cext, Cabs, Cbck


# In[7]:


def shellCS(wl,N1,N2,N,mu1,mu2,a,b):
    wl /= 1e9
    Csca = npy.empty(len(wl), dtype=npy.float)
    Cext = npy.empty(len(wl), dtype=npy.float)
    Cbck = npy.empty(len(wl), dtype=npy.float)
    Cabs = npy.empty(len(wl), dtype=npy.float)
    for index in range(0,len(wl)-1,1):
        temp = shellCSu(wl[index]*(1e9),N1[index],N2[index],N,mu1,mu2,a,b)
        Csca[index] = temp[0]*(1e18)
        Cext[index] = temp[1]*(1e18)
        Cabs[index] = temp[2]*(1e18)
        Cbck[index] = temp[3]*(1e18)
    return Csca, Cext, Cabs, Cbck

