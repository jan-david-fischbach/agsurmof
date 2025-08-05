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
from qnmsc.materials import to_omega, eps_surmof, eps_ag
from scipy.constants import c as c0, hbar, e
import numpy as np

@np.vectorize
def det_tmat(
  hbar_omega,
  materials = None,
  radius = 0.04,
  mmax = 0,
  beta = 0.7,
  thickness_surmof = 0.02,
  npoles = 3,
  scale_osc=1, 
  scale_damping=1
  ):

  if materials is None:
    materials = [
      treams.Material(eps_ag(hbar_omega)), 
      treams.Material(
        eps_surmof(
          hbar_omega, npoles, 
          scale_osc=scale_osc, scale_damping=scale_damping
        )), 
      treams.Material()
    ]
  
  k0 = to_omega(hbar_omega)/c0 * 1e-6 # in 1/um
  kz = k0 / beta
  T = treams.TMatrixC.cylinder(kzs=kz, mmax=mmax, k0=k0, radii=[radius, radius+thickness_surmof], materials=materials)
  return np.linalg.det(np.array(T))


# %%
treams.config.set_BRANCH_CUT_SQRT_MIE_N(-0.5*np.pi)

# %%
if __name__=='__main__':
  # Testing complex frequency calculations.
  import matplotlib.pyplot as plt

  domain = np.array([1.4-0.03j, 2+0.01j])
  res = 51
  hbar_omega = (
    np.linspace(domain[0].imag, domain[1].imag, num = res)[:, None]*1j + 
    np.linspace(domain[0].real, domain[1].real, num = res)
  )
  f = det_tmat(hbar_omega=hbar_omega, radius=0.04)

  # %%
  import diffaaable
  s = np.s_[::2, ::2]
  fit = diffaaable.aaa(hbar_omega[s], f[s])

  # %%
  plt.figure()
  plt.pcolormesh(
    hbar_omega.real, hbar_omega.imag, np.abs(f), 
    norm="log", rasterized=True,
  )
  plt.colorbar(label="$|\mathbf{T}|$")
  plt.scatter(fit[-1].real, fit[-1].imag, marker="x", color="C1")
  plt.xlabel("$\Re\{\hbar\omega\}$ [eV]")
  plt.ylabel("$\Im\{\hbar\omega\}$ [eV]")
  plt.xlim(domain.real)
  plt.ylim(domain.imag)

  plt.savefig("out/cylinder_beta.pdf", dpi=900)

# %%
