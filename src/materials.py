import jax.numpy as np
from scipy.constants import hbar, e

def to_omega(hbar_omega):
    return hbar_omega * e / hbar#in Hz

def _f_to_omega(f_Thz):
    return f_Thz * 2*np.pi*1e12

def eps_surmof(hbar_omega, npoles, scale_osc=1, scale_damping=1):
    """Relative Permittivity of the SURMOF material

    Args:
        hbar_omega (complex): in eV
        npoles (int): number of poles considered (between 1 and 3)
        scale_osc (int, optional): 
            Scaling factor for the oscillator strength. Defaults to 1.
        scale_damping (int, optional): 
            Scaling factor for the damping. Defaults to 1.

    Returns:
        complex: relative permittivity
    """
    
    # Material data
    f0 = np.array([412.93727009075883, 438.2930673770547, 448.79110491874115])  #in THz
    damping = np.array([5.3, 6.0, 6.2]) * scale_damping                         #in THz
    intensity = np.array([14.84804589, 1.63227866, 1.04500718]) * scale_osc
    eps_background = 1.6

    gamma   = _f_to_omega(damping[:npoles])
    omega_0 = _f_to_omega(f0[:npoles])
    omega_p = np.sqrt(intensity[:npoles]*gamma*omega_0)

    omega = to_omega(hbar_omega)
    eps = eps_background*np.ones_like(hbar_omega, dtype='complex')
    for i in range(npoles):
        eps += omega_p[i]**2 / (omega_0[i]**2 - omega**2 - 1j*omega*gamma[i])

    return eps

def eps_ag(hbar_omega):
    """Relative Permittivity of the silver material

    Args:
        hbar_omega (complex): in eV

    Returns:
        complex: relative permittivity
    """

    # Also for silver we have parameters from fitting.
    # These are from fp^2 / [f0^2 - f^2 - i f g] with normal frequencies in THz.
    Agw0    = _f_to_omega(134.39519365)
    Aggamma = _f_to_omega(10.61865288)
    Agwp    = _f_to_omega(1867.27147966)

    omega = to_omega(hbar_omega)
    return 1 + Agwp**2 / (Agw0**2 - omega**2 - 1j*Aggamma*omega)