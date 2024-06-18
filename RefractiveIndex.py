# %%
import math
import numpy as npy
from scipy.interpolate import interp1d

# %%
def Leff(a,b,isShell):
    return{
        False: 4*a/3,
        #False: a,
        #True: b-a,
        True: 4*(b**3 - a**3)/(b**2),
    }[isShell]

# %%
def q_density(shell_material):
        return{
            'Ag': 5.86e28,
            'Au': 5.90e28,
        }[shell_material]

# %%
def v_f(shell_material):
        return{
            'Ag': 1.39e6,
            'Au': 1.40e6,
        }[shell_material]

# %%
def gamma_0(shell_material):
            return{
                'Ag': 3.22e13,
                'Au': 1.07e14,
            }[shell_material]

# %%
def material_out(shell_material):
    return{
        'Ag': 'Ag_Johnson.txt',
        'Al': 'Al_McPeak.txt',
        'Au': 'Au_Johnson.txt',
        'Cu': 'Cu_Johnson.txt',
        'W': 'W_Werner.txt',
        'Si': 'Si_Schinke.txt',
        'PS': 'Polystyrene.txt'
    }[shell_material]

# %%
def medium(material_medium):
    return{
        'Water': 1.33,
        'Air': 1.00,
    }[material_medium]

# %%
def setupRI(shell_material,core_material,medium_material,a,b,lambda_min,lambda_max,isShell,drude):
    a /= 1e9
    b /= 1e9
    
    if type(medium_material) is str:
        N = medium(medium_material)
    else:
        N = medium_material
    
    wl2 = (1e-6)*npy.loadtxt('Materials/n_'+material_out(shell_material),usecols=0)
    n2_np = npy.loadtxt('Materials/n_'+material_out(shell_material),usecols=1)
    if shell_material == 'PS':
        k2_np = 0*npy.ones(len(wl2), dtype=float)
    else:
        k2_np = npy.loadtxt('Materials/k_'+material_out(shell_material),usecols=1)
    
    if isShell:
        if core_material == 'Silica':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_SiO2_Malitson.txt',usecols=0)
            n1_np = npy.loadtxt('Materials/n_SiO2_Malitson.txt',usecols=1)
            k1_np = 0*npy.ones(len(wl1), dtype=float)
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
        elif core_material == 'Water':
            wl1 = wl2
            n1_np = 1.33*npy.ones(len(wl1), dtype=float)
            k1_np = 0*npy.ones(len(wl1), dtype=float)
        elif core_material == 'Air':
            wl1 = wl2
            n1_np = 1*npy.ones(len(wl1), dtype=float)
            k1_np = 0*npy.ones(len(wl1), dtype=float)
        elif core_material == 'PS':
            wl1 = (1e-6)*npy.loadtxt('Materials/n_Polystyrene.txt', usecols=0)
            n1_np = npy.loadtxt('Materials/n_Polystyrene.txt', usecols=1)
            k1_np = 0*npy.ones(len(wl1), dtype=float)
    else:
        wl1 = wl2
        n1_np = 0*npy.ones(len(wl1), dtype=float)
        k1_np = 0*npy.ones(len(wl1), dtype=float)
    
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
    
    N1 = (fn1_np(wl) + fk1_np(wl)*(1.0j))*npy.sqrt(mu1)
    
    # Drude correction
    if drude:
        N2_pre = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)
        eps0 = 8.8541878128e-12
        q_e = -1.60217662e-19
        m_e = 9.10938356e-31
        omega_p = npy.sqrt(q_density(shell_material)*q_e*q_e/(eps0*m_e))
        A = 1
        omega = 2*math.pi*(3e8)/wl
        eps_intra = omega_p*omega_p/(omega*omega + (1.0j)*omega*gamma_0(shell_material))
        eps_corr = omega_p*omega_p/(omega*omega + (1.0j)*omega*(gamma_0(shell_material)+A*v_f(shell_material)/Leff(a,b,isShell)))
        N2 = npy.sqrt(N2_pre*N2_pre + eps_intra - eps_corr)
    else:
        N2 = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)    #Uncomment to disable drude correction
    
    return wl, N, N1, N2, mu1, mu2

# %%
def setupRIu(shell_material,core_material,medium_material,a,b,wl,isShell,drude):
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
            k1_np = 0*npy.ones(len(wl1), dtype=float)
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
            n1_np = 1.33*npy.ones(len(wl1), dtype=float)
            k1_np = 0*npy.ones(len(wl1), dtype=float)
        elif core_material == 'Air':
            wl1 = wl2
            n1_np = 1*npy.ones(len(wl1), dtype=float)
            k1_np = 0*npy.ones(len(wl1), dtype=float)
    else:
        wl1 = wl2
        n1_np = 0*npy.ones(len(wl1), dtype=float)
        k1_np = 0*npy.ones(len(wl1), dtype=float)
    
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
    
    # Drude correction
    if drude:
        N2_pre = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)
        eps0 = 8.8541878128e-12
        q_e = -1.60217662e-19
        m_e = 9.10938356e-31
        omega_p = npy.sqrt(q_density(shell_material)*q_e*q_e/(eps0*m_e))
        A = 1
        omega = 2*math.pi*(3e8)/wl
        eps_intra = omega_p*omega_p/(omega*omega + (1.0j)*omega*gamma_0(shell_material))
        eps_corr = omega_p*omega_p/(omega*omega + (1.0j)*omega*(gamma_0(shell_material)+A*v_f(shell_material)/Leff(a,b,isShell)))
        N2 = npy.sqrt(N2_pre*N2_pre + eps_intra - eps_corr)
    else:
        N2 = (fn2_np(wl) + fk2_np(wl)*(1.0j))*npy.sqrt(mu2)    #Uncomment to disable drude correction

    return N, N1, N2, mu1, mu2



# %%
