import jax.numpy as np
from scipy.constants import hbar, e

def to_omega(hbar_omega):
    return hbar_omega * e / hbar  # in Hz

def to_eV(omega):
    return omega * hbar / e

def _f_to_omega(f_Thz):
    return f_Thz * 2 * np.pi * 1e12

def surmof_material_data(scale_damping, scale_osc):
    omega_0 = _f_to_omega(
        np.array([412.93727009075883, 438.2930673770547, 448.79110491874115])
    )  # in s^-1

    gamma = _f_to_omega(
        np.array([5.3, 6.0, 6.2]) * scale_damping
    )  # in s^-1

    intensity = np.array([14.84804589, 1.63227866, 1.04500718]) * scale_osc
    eps_background = 1.6
    return omega_0, gamma, intensity, eps_background

def eps_surmof(hbar_omega, npoles, scale_osc=1, scale_damping=1):
    mat = surmof_material_data(scale_damping, scale_osc)
    omega_0, gamma, intensity, eps_background = mat

    gamma = gamma[:npoles]
    omega_0 = omega_0[:npoles]
    omega_p = np.sqrt(intensity[:npoles] * gamma * omega_0)

    omega = to_omega(hbar_omega)
    eps = eps_background * np.ones_like(hbar_omega, dtype='complex')
    for i in range(npoles):
        eps += omega_p[i]**2 / (omega_0[i]**2 - omega**2 - 1j * omega * gamma[i])

    return eps

def eps_ag(hbar_omega):
    Agw0 = _f_to_omega(134.39519365)
    Aggamma = _f_to_omega(10.61865288)
    Agwp = _f_to_omega(1867.27147966)

    omega = to_omega(hbar_omega)
    return 1 + Agwp**2 / (Agw0**2 - omega**2 - 1j * Aggamma * omega)

# ==============================
# Main part — runs when you hit "Run"
# ==============================

def main():
    hw = np.linspace(1, 2.5, 200)

    # Calculate values
    eps_ag_values = eps_ag(hw)
    eps_surmof_values = eps_surmof(hw, npoles=3)

    # Save eps_ag values
    with open("eps_ag.txt", "w") as f_ag:
        f_ag.write("# hbar_omega (eV)    eps_ag_real    eps_ag_imag\n")
        for i in range(len(hw)):
            f_ag.write(f"{hw[i]:.6f}    {eps_ag_values[i].real:.6e}    {eps_ag_values[i].imag:.6e}\n")

    # Save eps_surmof values
    with open("eps_surmof.txt", "w") as f_surmof:
        f_surmof.write("# hbar_omega (eV)    eps_surmof_real    eps_surmof_imag\n")
        for i in range(len(hw)):
            f_surmof.write(f"{hw[i]:.6f}    {eps_surmof_values[i].real:.6e}    {eps_surmof_values[i].imag:.6e}\n")

    print("✅ Files saved successfully: eps_ag.txt and eps_surmof.txt")

# This makes sure it only runs when you press "Run"
if __name__ == "__main__":
    main()
