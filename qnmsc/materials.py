import jax.numpy as np
from scipy.constants import hbar, e

def to_omega(hbar_omega):
    return hbar_omega * e / hbar#in Hz

def to_eV(omega):
    return omega * hbar / e

def _f_to_omega(f_Thz):
    return f_Thz * 2*np.pi*1e12

def surmof_material_data(scale_damping, scale_osc):
    omega_0 = _f_to_omega(
        np.array([412.93727009075883, 438.2930673770547, 448.79110491874115])
    ) # in s^-1

    gamma = _f_to_omega(
        np.array([5.3, 6.0, 6.2]) * scale_damping
    ) # in s^-1
    
    intensity = np.array([14.84804589, 1.63227866, 1.04500718]) * scale_osc
    eps_background = 1.6
    return omega_0, gamma, intensity, eps_background

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
    mat = surmof_material_data(scale_damping, scale_osc)
    omega_0, gamma, intensity, eps_background = mat

    gamma   = gamma[:npoles]
    omega_0 = omega_0[:npoles]
    omega_p = np.sqrt(intensity[:npoles]*gamma*omega_0)

    omega = to_omega(hbar_omega)
    eps = eps_background*np.ones_like(hbar_omega, dtype='complex')
    for i in range(npoles):
        eps += omega_p[i]**2 / (omega_0[i]**2 - omega**2 - 1j*omega*gamma[i])

    return eps

def eps_ag_markus(hbar_omega):
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

def eps_ag_sergei(hbar_omega):
    """Relative Permittivity of the silver material

    Args:
        hbar_omega (complex): in eV

    Returns:
        complex: relative permittivity
    """
    from scipy.constants import c as c0
    omega = to_omega(hbar_omega)/(c0*1e6)
    print(omega)

    poles = [0, -0.219035578322267j]
    residues = [9.055135084746365e+03j, -9.055135084746365e+03j]

    epsilon_hat = 4.608535749688730

    for pol, res in zip(poles, residues):
        epsilon_hat = epsilon_hat + res/(omega - pol)
    
    return epsilon_hat

if __name__ == "__main__":
    # Testing the eps_surmof function
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    hbar_omega = np.linspace(1.4, 2.3, 100) + 0.01j
    eps = eps_surmof(hbar_omega, npoles=3, scale_osc=1, scale_damping=1)
    eps_ag_m = eps_ag_markus(hbar_omega)
    eps_ag_s = eps_ag_sergei(hbar_omega)

    # plt.plot(hbar_omega.real, eps.real, label='Real part')
    # plt.plot(hbar_omega.real, eps.imag, label='Imaginary part')

    plt.plot(hbar_omega.real, eps_ag_m.real, label='Real part (Ag Markus)')
    plt.plot(hbar_omega.real, eps_ag_m.imag, label='Imaginary part (Ag Markus)')

    plt.plot(hbar_omega.real, eps_ag_s.real, label='Real part (Ag Sergei)')
    plt.plot(hbar_omega.real, eps_ag_s.imag, label='Imaginary part (Ag Sergei)')
    plt.xlabel('hbar omega (eV)')
    plt.ylabel('Relative Permittivity')
    plt.legend()
    plt.savefig("out/eps_compare.png")