'''
Complex-frequency transmission or reflection by a cavity
  (air) (silver) (molecular material) (silver) (air)
We do everything at normal incidence.
The example here is based on
  https://onlinelibrary.wiley.com/doi/epdf/10.1002/adma.202200350
'''

'''
Markus Nyman
2014
'''

import numpy as np
import treams
import treams.coeffs
import treams.special
import mie_smat

def fresnel(eps1, eps2):
    n1 = np.sqrt(eps1)
    n2 = np.sqrt(eps2)
    return 2*n1/(n1+n2), 2*n2/(n1+n2), (n1-n2)/(n1+n2), (n2-n1)/(n1+n2)

def slab_trref(wfreq, d, eps1, eps2, eps3):
    tau12, tau21, rho12, rho21 = fresnel(eps1, eps2)
    tau23, tau32, rho23, rho32 = fresnel(eps2, eps3)
    k2 = wfreq*np.sqrt(eps2)
    den = 1 - rho21*rho23*np.exp(1j*k2*2*d)
    tr = tau12*tau23*np.exp(1j*k2*d) / den
    ref = rho12 + tau12*tau21*rho23*np.exp(1j*k2*2*d) / den
    return tr, ref

def threelayerstack_trref(wfreq, d2, d3, d4, eps1, eps2, eps3, eps4, eps5):
    # Forward direction 1 2 3 4 5
    m1tr_fwd, m1ref_fwd = slab_trref(wfreq, d2, eps1, eps2, eps3)
    m2tr_fwd, m2ref_fwd = slab_trref(wfreq, d4, eps3, eps4, eps5)
    # Backward direction 5 4 3 2 1
    m1tr_back, m1ref_back = slab_trref(wfreq, d2, eps3, eps2, eps1)
    m2tr_back, m2ref_back = slab_trref(wfreq, d4, eps5, eps4, eps3)
    # Whole stack
    k3 = wfreq*np.sqrt(eps3)
    den = 1 - m1ref_back * m2ref_fwd * np.exp(1j*k3*2*d3)
    tr = m1tr_fwd * m2tr_fwd * np.exp(1j*k3*d3) / den
    ref = m1ref_fwd + m1tr_fwd * m1tr_back * m2ref_fwd * np.exp(1j*k3*2*d3) / den
    return tr, ref
    
def eps_cav(wfreq, npoles):
    f0 = np.array([448.79110491874115, 438.2930673770547, 412.93727009075883])
    intensity = np.array([1.04500718, 1.63227866, 14.84804589])
    damping = np.array([6.2, 6.0, 5.3])

    f0 = f0[-npoles:]
    intensity = intensity[-npoles:]
    damping = damping[-npoles:]

    gamma = damping*2*np.pi
    w0 = 2*np.pi*f0
    wp = np.sqrt(intensity*gamma*w0)
    # These frequencies are now in THz scale. For numerical sanity, we're going to be unitless;
    #   freq = 1 is 300 THz
    #   length = 1 is 1 micrometer
    # Speed of light is now 3e8 m/s which is good enough.
    w0 /= 300
    wp /= 300
    gamma /= 300

    eps_background = 1.6

    eps = eps_background*np.ones(wfreq.shape, dtype='complex')
    for i in range(npoles):
        eps += wp[i]**2 / (w0[i]**2 - wfreq**2 - 1j*wfreq*gamma[i])

    return eps

def ag_surmof_cavity_trref(wfreq, thickness, npoles:int = 3): #Previously used
    # Material data
    # See verify_data.py for explanation on how to transform these to the usual quantities.

    # Also for silver we have parameters from fitting.
    # These are from fp^2 / [f0^2 - f^2 - i f g] with normal frequencies in THz.
    Agw0 = 2*np.pi*134.39519365 / 300
    Aggamma = 2*np.pi*10.61865288 / 300
    Agwp = 2*np.pi*1867.27147966 / 300

    # Mirror thicknesses, these are taken from Benedikt's paper
    mirror1d = 0.01
    mirror2d = 0.03

    eps_air = 1
    eps_Ag = 1 + Agwp**2 / (Agw0**2 - wfreq**2 - 1j*Aggamma*wfreq)
    eps_cavity = eps_cav(wfreq, npoles)

    #return np.sqrt(eps_cav)
    tr, ref = threelayerstack_trref(wfreq, mirror1d, thickness, mirror2d, eps_air, eps_Ag, eps_cavity, eps_Ag, eps_air)
    
    # Here, we can decide what quantity to return.
    return tr#, tr
    #return tr, ref  # this can be used for testing

def ag_surmof_cavity_det_smat(wfreq, thickness, npoles:int = 3):

    # Mirror thicknesses, these are taken from Benedikt's paper
    mirror1d = 0.01
    mirror2d = 0.03

    eps_air = 1
    eps_Ag = 4.60853575 + 9055.04799147j * (1/(wfreq) - 1/(wfreq+0.21903558j))
    eps_cavity = eps_cav(wfreq, npoles)

    #return np.sqrt(eps_cav)
    tr, ref = threelayerstack_trref(wfreq, mirror1d, thickness, mirror2d, eps_air, eps_Ag, eps_cavity, eps_Ag, eps_air)
    tr, ref2 = threelayerstack_trref(wfreq, mirror2d, thickness, mirror1d, eps_air, eps_Ag, eps_cavity, eps_Ag, eps_air)
    
    # Here, we can decide what quantity to return.
    S = np.moveaxis([[ref, tr],[tr, ref2]], -1, 0)
    print(S.shape)
    return np.linalg.det(S)


def ag_surmof_core_shell(wfreq, d_core, npoles:int = 3, l=1):

    mirror = 0.02 # Average mirror thickness from Benedikt's paper

    eps_air = 1
    eps_Ag = 4.60853575 + 9055.04799147j * (1/(wfreq) - 1/(wfreq+0.21903558j))
    eps_cavity = eps_cav(wfreq, npoles)

    k0 = wfreq

    coeffs = [treams.coeffs.mie(l, k * np.array([d_core/2, d_core/2+mirror]), [eps_c, eps_a, eps_air], [1,1,1], [0,0,0])[0,0] 
              for k, eps_c, eps_a in zip(k0, eps_cavity, eps_Ag)]

    #coeffs = [treams.coeffs.mie(l,  [k*d_core/2], [eps_c, eps_air], [1,1], [0,0])[0,0] for k, eps_c in zip(k0, eps_cavity)]

    return np.array(coeffs)

if __name__=='__main__':
    # Testing complex frequency calculations.
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from scipy.io import loadmat
    import os.path

    # 7.0804 10.1148
    cache_file = "Tr.npy"

    wfreqr = np.linspace(7.0804, 10.1148, 2000)
    wfreqi = np.linspace(-0.5, 0.5, 2000)

    wfreqr = np.linspace(8.55, 8.66, 200)
    wfreqi = np.linspace(-0.0, -0.1, 200)


    Wfreqr, Wfreqi = np.meshgrid(wfreqr, wfreqi, indexing='ij')
    Wfreq = Wfreqr + 1j*Wfreqi
    if os.path.isfile(cache_file):
        Tr = np.load(cache_file)
    else:
        Tr = ag_surmof_cavity_trref(Wfreq, 0.4)
        #Tr = 1/(1-(0.9**2*np.exp(2j*n*Wfreq*0.05)))
        #np.save(cache_file, Tr)

    #poles = loadmat("examples/poles.mat")["poles"]
    
    plt.figure()
    plt.pcolormesh(Wfreqr, Wfreqi, abs(Tr), norm="log", vmax=1e2, vmin=1e-4)
    plt.colorbar()
    #plt.scatter(poles.real, poles.imag, marker="x", linewidths=0.2, color="red")
    plt.xlabel("$\Re\{\omega\}$")
    plt.ylabel("$\Im\{\omega\}$")
    plt.savefig("surmof_scan.png", dpi=900)