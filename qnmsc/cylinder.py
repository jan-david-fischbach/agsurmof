# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%

import treams
from qnmsc.materials import to_omega
from scipy.constants import c as c0, hbar, e
import numpy as np

@np.vectorize
def det_tmat(
  hbar_omega,
  kz,
  radius = 0.3,
  materials = None,
  mmax = 2
  ):

  if materials is None:
    materials = [treams.Material(12+1e-1j), treams.Material()]
  
  k0 = to_omega(hbar_omega)/c0 * 1e-6 # in 1/um
  T = treams.TMatrixC.cylinder(kzs=kz, mmax=mmax, k0=k0, radii=[radius], materials=materials)
  return np.linalg.det(np.array(T))


# %%
treams.config.set_BRANCH_CUT_SQRT_MIE_N(-0.5*np.pi)

# %%
if __name__=='__main__':
  # Testing complex frequency calculations.
  import matplotlib.pyplot as plt

  domain = np.array([1.4-0.03j, 2+0.01j])
  res = 301
  hbar_omega = (
    np.linspace(domain[0].imag, domain[1].imag, num = res)[:, None]*1j + 
    np.linspace(domain[0].real, domain[1].real, num = res)
  )
  kz = 9
  E_kz = kz*1e6*c0*hbar/e
  f = det_tmat(hbar_omega=hbar_omega, kz=kz)

  # %%
  import diffaaable
  s = np.s_[::20, ::20]
  fit = diffaaable.aaa(hbar_omega[s], f[s])

  # %%
  plt.figure()
  plt.pcolormesh(
    hbar_omega.real, hbar_omega.imag, np.abs(f), 
    norm="log", rasterized=True, vmin=1e-11
  )
  plt.colorbar(label="$|\mathbf{T}|$")
  plt.scatter([E_kz], [0], marker="x")
  plt.scatter(fit[-1].real, fit[-1].imag, marker="x")
  plt.xlabel("$\Re\{\hbar\omega\}$ [eV]")
  plt.ylabel("$\Im\{\hbar\omega\}$ [eV]")
  plt.xlim(domain.real)
  plt.ylim(domain.imag)

  plt.title(f"$k_z = {kz} \mu m^{{-1}}$ ({E_kz:.3f}eV)")

  plt.savefig("cylinder.pdf", dpi=900)

# %%
