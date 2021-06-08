#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as npy
import RefractiveIndex as ri
from scipy import special as sp
import opticalCrossSection as ocs


# In[2]:


def sphereEMu(wl,N2,N,r,theta,phi,a,mu1):
    # Importação dos coeficientes ópticos
    a_n, b_n, c_n, d_n = ocs.sphereCoeffs(wl,N2,N,mu1,a)
    wl /= 1e9
    r /= 1e9
    a /= 1e9
    omega = 2*math.pi*(3e8)/wl
    k = (2*math.pi*N)/wl                                    # número de onda do meio
    k1 = (2*math.pi*N2)/wl
    x = k*a                                                 # parâmetro de tamanho (esfera)
    v_max = math.ceil(x + 4.05*x**(1/3) + 2)                # ordem máxima necessária (esfera)
    # Parâmetros gerais e constantes
    mu0 = 4*math.pi*(1e-7)                                               # Permeabilidade magnética do vácuo
    E_0 = 1                                                              # Magnitude do campo elétrico incidente (V/m)
    E_n = npy.empty(v_max, dtype=npy.cfloat)                             # Constante de ordem do campo elétrico
    pi_n = npy.empty(v_max, dtype=npy.cfloat)                            # Função de dependência angular 1
    tau_n = npy.empty(v_max, dtype=npy.cfloat)                           # Função de dependência angular 2
    Mo1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Mo1n^(p)
    Mo1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Mo1n^(p)
    Me1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Me1n^(p)
    Me1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Me1n^(p)
    No1np_r = npy.empty(v_max, dtype=npy.cfloat)                         # Componente r No1n^(p)
    No1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta No1n^(p)
    No1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi No1n^(p)
    Ne1np_r = npy.empty(v_max, dtype=npy.cfloat)                         # Componente r Ne1n^(p)
    Ne1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Ne1n^(p)
    Ne1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Ne1n^(p)
    # Cálculo dos parâmetros
    E_n[0] = E_0
    pi_n[0] = 0
    pi_n[1] = 1
    for v in range(1,v_max,1):
        E_n[v] = ((1.0j)**v)*((2*v+1)/(v*(v+1)))*E_0
        if v>1:
            pi_n[v] = ((2*v-1)/(v-1))*npy.cos(theta)*pi_n[v-1] - v/(v-1)*pi_n[v-2]
        tau_n[v] = v*npy.cos(theta)*pi_n[v] - (v+1)*pi_n[v-1]
    for v in range(1,v_max,1):
        if r<a:                                                          # Para r dentro da NP (campo interno)
            Mo1np_theta[v] = npy.cos(phi)*pi_n[v]*sp.spherical_jn(v,k1*r)
            Mo1np_phi[v] = -npy.sin(phi)*tau_n[v]*sp.spherical_jn(v,k1*r)
            Me1np_theta[v] = -npy.sin(phi)*pi_n[v]*sp.spherical_jn(v,k1*r)
            Me1np_phi[v] = -npy.cos(phi)*tau_n[v]*sp.spherical_jn(v,k1*r)
            No1np_r[v] = npy.sin(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k1*r)/(k1*r))
            No1np_theta[v] = npy.sin(phi)*tau_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r))
            No1np_phi[v] = npy.cos(phi)*pi_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r))
            Ne1np_r[v] = npy.cos(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k1*r)/(k1*r))
            Ne1np_theta[v] = npy.cos(phi)*tau_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r))
            Ne1np_phi[v] = -npy.sin(phi)*pi_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r)) 
        else:                                                            # Para r fora da NP (campo espalhado)
            Mo1np_theta[v] = npy.cos(phi)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
            Mo1np_phi[v] = -npy.sin(phi)*tau_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
            Me1np_theta[v] = -npy.sin(phi)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
            Me1np_phi[v] = -npy.cos(phi)*tau_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
            No1np_r[v] = npy.sin(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*(sp.spherical_yn(v,k*r))/(k*r))
            No1np_theta[v] = npy.sin(phi)*tau_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
            No1np_phi[v] = npy.cos(phi)*pi_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
            Ne1np_r[v] = npy.cos(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*(sp.spherical_yn(v,k*r))/(k*r))
            Ne1np_theta[v] = npy.cos(phi)*tau_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
            Ne1np_phi[v] = -npy.sin(phi)*pi_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
    # Cálculo dos campos
    E_r = 0
    E_theta = 0
    E_phi = 0
    H_r = 0
    H_theta = 0
    H_phi = 0
    if r<a:                                                              # Para r dentro da NP (campo interno)
        for v in range(1,v_max,1):
            E_r += -(1.0j)*E_n[v]*d_n[v]*Ne1np_r[v]
            E_theta += E_n[v]*(c_n[v]*Mo1np_theta[v]-(1.0j)*d_n[v]*Ne1np_theta[v])
            E_phi += E_n[v]*(c_n[v]*Mo1np_phi[v]-(1.0j)*d_n[v]*Ne1np_phi[v])
            H_r += -(1.0j)*k1*E_n[v]*c_n[v]*No1np_r[v]/(omega*mu1*mu0)
            H_theta += -k1*E_n[v]*(d_n[v]*Me1np_theta[v]+(1.0j)*c_n[v]*No1np_theta[v])/(omega*mu1*mu0)
            H_phi += -k1*E_n[v]*(d_n[v]*Me1np_phi[v]+(1.0j)*c_n[v]*No1np_phi[v])/(omega*mu1*mu0)
    else:                                                                # Para r fora da NP (campo espalhado)
        for v in range(1,v_max,1):
            E_r += (1.0j)*E_n[v]*a_n[v]*Ne1np_r[v]
            E_theta += E_n[v]*((1.0j)*a_n[v]*Ne1np_theta[v]-b_n[v]*Mo1np_theta[v])
            E_phi += E_n[v]*((1.0j)*a_n[v]*Ne1np_phi[v]-b_n[v]*Mo1np_phi[v])
            H_r += (1.0j)*k*E_n[v]*b_n[v]*No1np_r[v]/(omega*mu0)
            H_theta += k*E_n[v]*((1.0j)*b_n[v]*No1np_theta[v]+a_n[v]*Me1np_theta[v])/(omega*mu0)
            H_phi += k*E_n[v]*((1.0j)*b_n[v]*No1np_phi[v]+a_n[v]*Me1np_phi[v])/(omega*mu0)
    # Estruturação dos vetores de campo
    E = npy.array([E_r, E_theta, E_phi])
    H = npy.array([H_r, H_theta, H_phi])
    return E, H


# In[3]:


def shellEMu(wl,N2,N1,N,r,theta,phi,a,b,mu1,mu2):
    # Importação dos coeficientes ópticos
    a_n, b_n, c_n, d_n, f_n, g_n, v_n, w_n = ocs.shellCoeffs(wl,N1,N2,N,mu1,mu2,a,b)
    wl /= 1e9
    r /= 1e9
    a /= 1e9
    b /= 1e9
    omega = 2*math.pi*(3e8)/wl
    k = (2*math.pi*N)/wl                                    # número de onda do meio
    k1 = (2*math.pi*N1)/wl
    k2 = (2*math.pi*N2)/wl
    y = k*b                                                 # parâmetro de tamanho (esfera)
    v_max = math.ceil(y + 4.05*y**(1/3) + 2)                # ordem máxima necessária (esfera)
    # Parâmetros gerais e constantes
    mu0 = 4*math.pi*(1e-7)                                               # Permeabilidade magnética do vácuo
    E_0 = 1                                                              # Magnitude do campo elétrico incidente (V/m)
    E_n = npy.empty(v_max, dtype=npy.cfloat)                             # Constante de ordem do campo elétrico
    pi_n = npy.empty(v_max, dtype=npy.cfloat)                            # Função de dependência angular 1
    tau_n = npy.empty(v_max, dtype=npy.cfloat)                           # Função de dependência angular 2
    Mo1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Mo1n^(p)
    Mo1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Mo1n^(p)
    Me1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Me1n^(p)
    Me1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Me1n^(p)
    No1np_r = npy.empty(v_max, dtype=npy.cfloat)                         # Componente r No1n^(p)
    No1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta No1n^(p)
    No1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi No1n^(p)
    Ne1np_r = npy.empty(v_max, dtype=npy.cfloat)                         # Componente r Ne1n^(p)
    Ne1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Ne1n^(p)
    Ne1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Ne1n^(p)
    # Cálculo dos parâmetros
    E_n[0] = E_0
    pi_n[0] = 0
    pi_n[1] = 1
    for v in range(1,v_max,1):
        E_n[v] = ((1.0j)**v)*((2*v+1)/(v*(v+1)))*E_0
        if v>1:
            pi_n[v] = ((2*v-1)/(v-1))*npy.cos(theta)*pi_n[v-1] - v/(v-1)*pi_n[v-2]
        tau_n[v] = v*npy.cos(theta)*pi_n[v] - (v+1)*pi_n[v-1]
    for v in range(1,v_max,1):
        if r<a:                                                          # Para r no núcleo (campo interno)
            Mo1np_theta[v] = npy.cos(phi)*pi_n[v]*sp.spherical_jn(v,k1*r)
            Mo1np_phi[v] = -npy.sin(phi)*tau_n[v]*sp.spherical_jn(v,k1*r)
            Me1np_theta[v] = -npy.sin(phi)*pi_n[v]*sp.spherical_jn(v,k1*r)
            Me1np_phi[v] = -npy.cos(phi)*tau_n[v]*sp.spherical_jn(v,k1*r)
            No1np_r[v] = npy.sin(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k1*r)/(k1*r))
            No1np_theta[v] = npy.sin(phi)*tau_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r))
            No1np_phi[v] = npy.cos(phi)*pi_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r))
            Ne1np_r[v] = npy.cos(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k1*r)/(k1*r))
            Ne1np_theta[v] = npy.cos(phi)*tau_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r))
            Ne1np_phi[v] = -npy.sin(phi)*pi_n[v]*(sp.spherical_jn(v-1,k1*r) - (v/(k1*r))*sp.spherical_jn(v,k1*r)) 
        else:                                                            # Para r na casca (campo interno)
            if r<b:
                Mo1n2_theta = npy.empty(v_max, dtype=npy.cfloat)         # Componente theta Mo1n^(2)
                Mo1n2_phi = npy.empty(v_max, dtype=npy.cfloat)           # Componente phi Mo1n^(2)
                Me1n2_theta = npy.empty(v_max, dtype=npy.cfloat)         # Componente theta Me1n^(2)
                Me1n2_phi = npy.empty(v_max, dtype=npy.cfloat)           # Componente phi Me1n^(2)
                No1n2_r = npy.empty(v_max, dtype=npy.cfloat)             # Componente r No1n^(2)
                No1n2_theta = npy.empty(v_max, dtype=npy.cfloat)         # Componente theta No1n^(2)
                No1n2_phi = npy.empty(v_max, dtype=npy.cfloat)           # Componente phi No1n^(2)
                Ne1n2_r = npy.empty(v_max, dtype=npy.cfloat)             # Componente r Ne1n^(2)
                Ne1n2_theta = npy.empty(v_max, dtype=npy.cfloat)         # Componente theta Ne1n^(2)
                Ne1n2_phi = npy.empty(v_max, dtype=npy.cfloat)           # Componente phi Ne1n^(2)
                Mo1np_theta[v] = npy.cos(phi)*pi_n[v]*sp.spherical_jn(v,k2*r)
                Mo1np_phi[v] = -npy.sin(phi)*tau_n[v]*sp.spherical_jn(v,k2*r)
                Me1np_theta[v] = -npy.sin(phi)*pi_n[v]*sp.spherical_jn(v,k2*r)
                Me1np_phi[v] = -npy.cos(phi)*tau_n[v]*sp.spherical_jn(v,k2*r)
                No1np_r[v] = npy.sin(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k2*r)/(k2*r))
                No1np_theta[v] = npy.sin(phi)*tau_n[v]*(sp.spherical_jn(v-1,k2*r) - (v/(k2*r))*sp.spherical_jn(v,k2*r))
                No1np_phi[v] = npy.cos(phi)*pi_n[v]*(sp.spherical_jn(v-1,k2*r) - (v/(k2*r))*sp.spherical_jn(v,k2*r))
                Ne1np_r[v] = npy.cos(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k2*r)/(k2*r))
                Ne1np_theta[v] = npy.cos(phi)*tau_n[v]*(sp.spherical_jn(v-1,k2*r) - (v/(k2*r))*sp.spherical_jn(v,k2*r))
                Ne1np_phi[v] = -npy.sin(phi)*pi_n[v]*(sp.spherical_jn(v-1,k2*r) - (v/(k2*r))*sp.spherical_jn(v,k2*r))
                Mo1n2_theta[v] = npy.cos(phi)*pi_n[v]*sp.spherical_yn(v,k2*r)
                Mo1n2_phi[v] = -npy.sin(phi)*tau_n[v]*sp.spherical_yn(v,k2*r)
                Me1n2_theta[v] = -npy.sin(phi)*pi_n[v]*sp.spherical_yn(v,k2*r)
                Me1n2_phi[v] = -npy.cos(phi)*tau_n[v]*sp.spherical_yn(v,k2*r)
                No1n2_r[v] = npy.sin(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_yn(v,k2*r)/(k2*r))
                No1n2_theta[v] = npy.sin(phi)*tau_n[v]*(sp.spherical_yn(v-1,k2*r) - (v/(k2*r))*sp.spherical_yn(v,k2*r))
                No1n2_phi[v] = npy.cos(phi)*pi_n[v]*(sp.spherical_yn(v-1,k2*r) - (v/(k2*r))*sp.spherical_yn(v,k2*r))
                Ne1n2_r[v] = npy.cos(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_yn(v,k2*r)/(k2*r))
                Ne1n2_theta[v] = npy.cos(phi)*tau_n[v]*(sp.spherical_yn(v-1,k2*r) - (v/(k2*r))*sp.spherical_yn(v,k2*r))
                Ne1n2_phi[v] = -npy.sin(phi)*pi_n[v]*(sp.spherical_yn(v-1,k2*r) - (v/(k2*r))*sp.spherical_yn(v,k2*r))
            else:                                                        # Para r fora da NP (campo espalhado)
                Mo1np_theta[v] = npy.cos(phi)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
                Mo1np_phi[v] = -npy.sin(phi)*tau_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
                Me1np_theta[v] = -npy.sin(phi)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
                Me1np_phi[v] = -npy.cos(phi)*tau_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r))
                No1np_r[v] = npy.sin(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*(sp.spherical_yn(v,k*r))/(k*r))
                No1np_theta[v] = npy.sin(phi)*tau_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
                No1np_phi[v] = npy.cos(phi)*pi_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
                Ne1np_r[v] = npy.cos(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*(sp.spherical_jn(v,k*r)+(1.0j)*(sp.spherical_yn(v,k*r))/(k*r))
                Ne1np_theta[v] = npy.cos(phi)*tau_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
                Ne1np_phi[v] = -npy.sin(phi)*pi_n[v]*((sp.spherical_jn(v-1,k*r)+(1.0j)*sp.spherical_yn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)+(1.0j)*sp.spherical_yn(v,k*r)))
        
    # Cálculo dos campos
    E_r = 0
    E_theta = 0
    E_phi = 0
    H_r = 0
    H_theta = 0
    H_phi = 0
    if r<a:                                                              # Para r dentro da NP (campo interno)
        for v in range(1,v_max,1):
            E_r += -(1.0j)*E_n[v]*d_n[v]*Ne1np_r[v]
            E_theta += E_n[v]*(c_n[v]*Mo1np_theta[v]-(1.0j)*d_n[v]*Ne1np_theta[v])
            E_phi += E_n[v]*(c_n[v]*Mo1np_phi[v]-(1.0j)*d_n[v]*Ne1np_phi[v])
            H_r += -(1.0j)*k1*E_n[v]*c_n[v]*No1np_r[v]/(omega*mu1*mu0)
            H_theta += -k1*E_n[v]*(d_n[v]*Me1np_theta[v]+(1.0j)*c_n[v]*No1np_theta[v])/(omega*mu1*mu0)
            H_phi += -k1*E_n[v]*(d_n[v]*Me1np_phi[v]+(1.0j)*c_n[v]*No1np_phi[v])/(omega*mu1*mu0)
    else:                                                                # Para r fora da NP (campo espalhado)
        if r<b:
            for v in range(1,v_max,1):
                E_r += -(1.0j)*E_n[v]*(g_n[v]*Ne1np_r[v]+w_n[v]*Ne1n2_r[v])
                E_theta += E_n[v]*(f_n[v]*Mo1np_theta[v]-(1.0j)*g_n[v]*Ne1np_theta[v]+v_n[v]*Mo1n2_theta[v]-(1.0j)*w_n[v]*Ne1n2_theta[v])
                E_phi += E_n[v]*(f_n[v]*Mo1np_phi[v]-(1.0j)*g_n[v]*Ne1np_phi[v]+v_n[v]*Mo1n2_phi[v]-(1.0j)*w_n[v]*Ne1n2_phi[v])
                H_r += -(1.0j)*k2*E_n[v]*(f_n[v]*No1np_r[v]+v_n[v]*No1n2_r[v])/(omega*mu2*mu0)
                H_theta += -k2*E_n[v]*(g_n[v]*Me1np_theta[v]+(1.0j)*f_n[v]*No1np_theta[v]+w_n[v]*Me1n2_theta[v]+(1.0j)*v_n[v]*No1n2_theta[v])/(omega*mu2*mu0)
                H_phi += -k2*E_n[v]*(g_n[v]*Me1np_phi[v]+(1.0j)*f_n[v]*No1np_phi[v]+w_n[v]*Me1n2_phi[v]+(1.0j)*v_n[v]*No1n2_phi[v])/(omega*mu2*mu0)
        else:
            for v in range(1,v_max,1):
                E_r += (1.0j)*E_n[v]*a_n[v]*Ne1np_r[v]
                E_theta += E_n[v]*((1.0j)*a_n[v]*Ne1np_theta[v]-b_n[v]*Mo1np_theta[v])
                E_phi += E_n[v]*((1.0j)*a_n[v]*Ne1np_phi[v]-b_n[v]*Mo1np_phi[v])
                H_r += (1.0j)*k*E_n[v]*b_n[v]*No1np_r[v]/(omega*mu0)
                H_theta += k*E_n[v]*((1.0j)*b_n[v]*No1np_theta[v]+a_n[v]*Me1np_theta[v])/(omega*mu0)
                H_phi += k*E_n[v]*((1.0j)*b_n[v]*No1np_phi[v]+a_n[v]*Me1np_phi[v])/(omega*mu0)
    # Estruturação dos vetores de campo
    E = npy.array([E_r, E_theta, E_phi])
    H = npy.array([H_r, H_theta, H_phi])
    return E, H


# In[4]:


def incidentEMu(wl,N,r,theta,phi):
    wl /= 1e9
    r /= 1e9
    omega = 2*math.pi*(3e8)/wl
    k = (2*math.pi*N)/wl                                    # número de onda do meio
    v_max = 15                # ordem máxima necessária (esfera)
    # Parâmetros gerais e constantes
    mu0 = 4*math.pi*(1e-7)                                               # Permeabilidade magnética do vácuo
    E_0 = 1                                                              # Magnitude do campo elétrico incidente (V/m)
    E_n = npy.empty(v_max, dtype=npy.cfloat)                             # Constante de ordem do campo elétrico
    pi_n = npy.empty(v_max, dtype=npy.cfloat)                            # Função de dependência angular 1
    tau_n = npy.empty(v_max, dtype=npy.cfloat)                           # Função de dependência angular 2
    Mo1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Mo1n^(p)
    Mo1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Mo1n^(p)
    Me1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Me1n^(p)
    Me1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Me1n^(p)
    No1np_r = npy.empty(v_max, dtype=npy.cfloat)                         # Componente r No1n^(p)
    No1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta No1n^(p)
    No1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi No1n^(p)
    Ne1np_r = npy.empty(v_max, dtype=npy.cfloat)                         # Componente r Ne1n^(p)
    Ne1np_theta = npy.empty(v_max, dtype=npy.cfloat)                     # Componente theta Ne1n^(p)
    Ne1np_phi = npy.empty(v_max, dtype=npy.cfloat)                       # Componente phi Ne1n^(p)
    # Cálculo dos parâmetros
    E_n[0] = E_0
    pi_n[0] = 0
    pi_n[1] = 1
    for v in range(1,v_max,1):
        E_n[v] = ((1.0j)**v)*((2*v+1)/(v*(v+1)))*E_0
        if v>1:
            pi_n[v] = ((2*v-1)/(v-1))*npy.cos(theta)*pi_n[v-1] - v/(v-1)*pi_n[v-2]
        tau_n[v] = v*npy.cos(theta)*pi_n[v] - (v+1)*pi_n[v-1]
        Mo1np_theta[v] = npy.cos(phi)*pi_n[v]*(sp.spherical_jn(v,k*r))
        Mo1np_phi[v] = -npy.sin(phi)*tau_n[v]*(sp.spherical_jn(v,k*r))
        Me1np_theta[v] = -npy.sin(phi)*pi_n[v]*(sp.spherical_jn(v,k*r))
        Me1np_phi[v] = -npy.cos(phi)*tau_n[v]*(sp.spherical_jn(v,k*r))
        No1np_r[v] = npy.sin(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*((sp.spherical_jn(v,k*r))/(k*r))
        No1np_theta[v] = npy.sin(phi)*tau_n[v]*((sp.spherical_jn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)))
        No1np_phi[v] = npy.cos(phi)*pi_n[v]*((sp.spherical_jn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)))
        Ne1np_r[v] = npy.cos(phi)*v*(v+1)*npy.sin(theta)*pi_n[v]*((sp.spherical_jn(v,k*r))/(k*r))
        Ne1np_theta[v] = npy.cos(phi)*tau_n[v]*((sp.spherical_jn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)))
        Ne1np_phi[v] = -npy.sin(phi)*pi_n[v]*((sp.spherical_jn(v-1,k*r)) - (v/(k*r))*(sp.spherical_jn(v,k*r)))
    # Cálculo dos campos
    Ei_r = 0
    Ei_theta = 0
    Ei_phi = 0
    Hi_r = 0
    Hi_theta = 0
    Hi_phi = 0
    for v in range(1,v_max,1):
        Ei_r += -(1.0j)*E_n[v]*Ne1np_r[v]
        Ei_theta += E_n[v]*(Mo1np_theta[v]-(1.0j)*Ne1np_theta[v])
        Ei_phi += E_n[v]*(Mo1np_phi[v]-(1.0j)*Ne1np_phi[v])
        Hi_r += -(1.0j)*k*E_n[v]*No1np_r[v]/(omega*mu0)
        Hi_theta += -k*E_n[v]*(Me1np_theta[v]+(1.0j)*No1np_theta[v])/(omega*mu0)
        Hi_phi += -k*E_n[v]*(Me1np_phi[v]+(1.0j)*No1np_phi[v])/(omega*mu0)
    # Estruturação dos vetores de campo
    E = npy.array([Ei_r, Ei_theta, Ei_phi])
    H = npy.array([Hi_r, Hi_theta, Hi_phi])
    return E, H


# In[ ]:


def yzEM(r,theta,shell_material,core_material,medium_material,a,b,wl,isShell,drude):
    
    N, N1, N2, mu1, mu2 = ri.setupRIu(shell_material,core_material,medium_material,a,b,wl,isShell,drude)
    
    wl /= 1e9
    omega = 2*math.pi*(3e8)/wl
    phi = 0
    
    eps0 = 8.8541878128e-12
    
    E_abs = npy.empty([len(r),len(theta)], dtype=npy.float)
    H_abs = npy.empty([len(r),len(theta)], dtype=npy.float)
    E_i = npy.empty([len(r),len(theta)], dtype=npy.float)
    H_i = npy.empty([len(r),len(theta)], dtype=npy.float)
    q_abs = npy.empty([len(r),len(theta)], dtype=npy.float)
    
    if isShell:
        for p in range(0,len(r)-1,1):
            for q in range(0,len(theta)-1,1):
                E, H = shellEMu(wl*(1e9),N2,N1,N,r[p],theta[q],phi,a,b,mu1,mu2)
                Ei, Hi = incidentEMu(wl*(1e9),N,r[p],theta[q],phi)
                if r[p]<b:
                    E_abs[p,q] = npy.linalg.norm(E)
                    H_abs[p,q] = npy.linalg.norm(H)
                    if r[p]<a:
                        q_abs[p,q] = omega*eps0*((N1**2).imag)*(npy.linalg.norm(E)**2)/2
                    else:
                        q_abs[p,q] = omega*eps0*((N2**2).imag)*(npy.linalg.norm(E)**2)/2
                else:
                    E_abs[p,q] = npy.linalg.norm(Ei+E)
                    H_abs[p,q] = npy.linalg.norm(Hi+H)
                    q_abs[p,q] = 0
    else:
        for p in range(0,len(r)-1,1):
            for q in range(0,len(theta)-1,1):
                E, H = sphereEMu(wl*(1e9),N2,N,r[p],theta[q],phi,a,mu1)
                Ei, Hi = incidentEMu(wl*(1e9),N,r[p],theta[q],phi)
                if r[p]<a:
                    E_abs[p,q] = npy.linalg.norm(E)
                    H_abs[p,q] = npy.linalg.norm(H)
                    q_abs[p,q] = omega*eps0*((N2**2).imag)*(npy.linalg.norm(E)**2)/2
                else:
                    E_abs[p,q] = npy.linalg.norm(Ei+E)
                    H_abs[p,q] = npy.linalg.norm(Hi+H)
                    q_abs[p,q] = 0
    
    return E_abs, H_abs, q_abs


# In[ ]:




