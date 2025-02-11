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

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
import jax.numpy as np
import staaax.angled_stratified
import sax
from qnmsc.materials import to_omega, eps_surmof, eps_ag
from scipy.constants import c as c0
import staaax

def ag_surmof_cavity_smat(
  hbar_omega, thickness, npoles:int = 3, 
  scale_osc:float=1, scale_damping:float=1, 
  mirror1d = 0.01, mirror2d = 0.03
  ):
    
  n_surmof = np.sqrt(eps_surmof(
      hbar_omega, 
      npoles=npoles, 
      scale_osc=scale_osc, 
      scale_damping=scale_damping
  ))

  n_ag = np.sqrt(eps_ag(hbar_omega))

  ds = [mirror1d, thickness, mirror2d] #in um
  ns = [1, n_ag, n_surmof, n_ag, 1]
  k0 = to_omega(hbar_omega)/c0 * 1e-6 # in 1/um
  # print(f"{k0=}")
  kx = 0
  stack, info = staaax.angled_stratified.stack_smat_kx(ds, ns, k0, kx, pol="p")
  return stack()

def ag_surmof_cavity_det_smat(
  hbar_omega, thickness, npoles:int = 3, 
  scale_osc:float=1, scale_damping:float=1, 
  mirror1d = 0.01, mirror2d = 0.03
  ):
    
  smat = ag_surmof_cavity_smat(
    hbar_omega, thickness, npoles, 
    scale_osc, scale_damping, 
    mirror1d, mirror2d
  )
  smat, portmap = sax.sdense(smat)
  return np.linalg.det(smat)

if __name__=='__main__':
  # Testing complex frequency calculations.
  import matplotlib.pyplot as plt

  def plot_cmplx_plane(domain, **kwargs):

    hbar_omega = (
      np.linspace(domain[0].imag, domain[1].imag, num = 201)[:, None]*1j + 
      np.linspace(domain[0].real, domain[1].real, num = 201)
    )

    f = ag_surmof_cavity_det_smat(hbar_omega=hbar_omega, thickness=0.08, npoles=3)

    plt.figure()
    plt.pcolormesh(
      hbar_omega.real, hbar_omega.imag, np.abs(f), 
      norm="log", **kwargs
    )
    plt.colorbar()
    plt.xlabel("$\Re\{\hbar\omega\}$ [eV]")
    plt.ylabel("$\Im\{\hbar\omega\}$ [eV]")
    

  plot_cmplx_plane([1.4-0.03j, 2+0.01j])
  plt.savefig("surmof_scan.png", dpi=900)
  plot_cmplx_plane([1.7-0.012j, 1.71-0.01j], vmin=1e-5)

# %%
  plot_cmplx_plane([1.7077-0.01096j, 1.70774-0.010958j], vmin=1e-5)
  plt.grid()