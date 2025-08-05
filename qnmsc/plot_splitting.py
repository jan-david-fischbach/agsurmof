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
from qnmsc.track_qnms import filename, track_qnms
from qnmsc.plot_trajectories import load_data, select_modes
from qnmsc import inverse_eigenproblem
from qnmsc.mpl_config import um, inv_um, mm
import numpy as np

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format='retina'
if __name__ == "__main__":
  import qnmsc.mpl_config
  qnmsc.mpl_config.config()

# %%
def plot_splitting(
  modenumber, npoles, scale_osc, scale_damping, domain, 
  inv_d=True, ylim=(1.2, 2.5), xlim=(1,12), color_rabi="r",
  axs=None, return_fit=False, n_interp=300,
  c_os          = (0.8, 0.8, 0.8),
  c_grid        = (0.8, 0.8, 0.8),
  c_fundamental = "gray",
  c_higher      = (0, 0, 0, 0.2),
  c_fit         = "k",
  c_font        = "k",
  c_mat         = ["C0", "C1", "C2"],
  lw_higher     = 1,
  force_legend  = False,
  legend_loc    = 'best',
  plot_dots     = True
  ):

  poles, residues, thickness, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )
  poles_tracked, residues_tracked = track_qnms(poles, residues)

  selection, modenumber = select_modes(npoles, scale_osc, scale_damping, domain, modenumber=modenumber)
  selected = poles_tracked[:, selection]

  splitting = np.abs(np.real(selected[:, 0]-selected[:, -1]))
    
  try:
    rabi_idx = np.nanargmin(splitting)
    f_rabi = np.nanmin(splitting)
    rabi_ok = True
  except ValueError as e:
    rabi_ok = False
  print(f"QNMs at rabi: {selected[rabi_idx, :]}")
  
  def populate_legend():
    plt.plot([],[], color=c_fundamental, label="fundamental QNMs")
    plt.plot([],[], color=c_higher, linewidth=lw_higher, label="higher order QNMs")
    plt.plot([],[], "--", color=c_os, zorder=5, label="'uncoupled' cavity mode")
    if plot_dots:
      plt.plot([],[], color=c_fit, linestyle="none", marker=".", label="coupling fit")

  param = 1/thickness if inv_d else thickness
  rabi_param = param[rabi_idx]
  if axs is None:
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(90*mm, 90*mm), height_ratios=[3,1])
    plt.sca(axs[0])
    populate_legend()
  else:
    fig = plt.gcf()
    plt.sca(axs[0])
    if force_legend:
      populate_legend()

  f_mode = selected.real
  plt.plot(param, f_mode, color=c_fundamental)

  Cs=[]
  om_os=[]
  evs_fit = []

  param_interp = np.linspace(min(param), max(param), n_interp)
  for i, p in enumerate(param_interp):
      interp_d = 1/p if inv_d else p
      evs = []
      for pole in selected.T:
        real = np.interp(interp_d, thickness, pole.real)
        imag = np.interp(interp_d, thickness, pole.imag)
        evs.append(real + 1j*imag)

      om_o, *couplings = inverse_eigenproblem.solve_inv_eig(
          evs, material_poles
      )
      Cs.append(couplings)
      om_os.append(om_o)
      evs_fit.append(inverse_eigenproblem.test_fwd_eig(om_o, material_poles, np.sqrt(couplings)))

  if npoles == 1:
    resonance_idx = np.nanargmin(np.abs(np.array(om_os).real - material_poles[0].real))
    print(f"QNMs at res: {np.array(evs_fit)[resonance_idx, :]}")

  if plot_dots:
    plt.plot(param_interp, np.array(evs_fit).real, color=c_fit, linestyle="none", marker=".")
  plt.plot(param_interp, np.array(om_os).real, "--", color=c_os, zorder=5)
  plt.vlines(
    [rabi_param],*f_mode[rabi_idx, [0, -1]], color=color_rabi, 
    label=f"$\Omega_\mathrm{{Rabi}}=2\cdot{f_rabi.real/2:.3f}[\mathrm{{eV}}]$",
    zorder=6
  )
  plt.ylabel("$\hbar \omega$ [eV]")
  plt.legend(fontsize=6, labelcolor=c_font, loc=legend_loc)

  for i,mat_pole in enumerate(material_poles):
      plt.axhline(mat_pole.real, color=c_mat[i],)#, linestyle="--")

  for pole in poles_tracked.T: # higher order modes
    plt.plot(param, pole.real, color=c_higher, linewidth=lw_higher, zorder=-1)

  plt.axvline(rabi_param, color=c_grid, zorder=-1)
  plt.ylim(ylim)
  plt.sca(axs[1])
  plt.axvline(rabi_param, color=c_grid, zorder=-1)

  plt.ylabel("Coupling $\sqrt{\hat g_i g_i}$ [eV]")
  Cs = np.array(Cs)
  cs = np.sqrt(np.real(Cs))
  for i,coupling in enumerate(cs.T):
      plt.plot(param_interp, coupling, ".-" if plot_dots else "-", color=c_mat[i], label=f"$p_{i+1}=\complexqty{{{material_poles[i]:.3f}}}{{eV}}$")

  gamma_avg = -np.sum(np.array(evs_fit), axis=-1).imag/(npoles+1) * cs[:,0]/cs[:,0]
  gamma_avg_res = np.interp(rabi_param, param_interp, gamma_avg)

  plt.plot(param_interp, gamma_avg, color="k", label=f"$\gamma_\mathrm{{avg}} = {gamma_avg_res:.3f}$ eV @ $\delta = 0$")
  
  stud = 0.1
  x = [rabi_param-stud,rabi_param+stud]
  plt.plot(x, [f_rabi.real/2]*2, color="r")
  #plt.ylim(0, 1.1*max(f_rabi.real/2, np.nanmax(cs.flatten())))

  plt.legend(fontsize=6, title="Material Resonances", loc="lower right", frameon=True)
  if inv_d:
      fig.supxlabel(r"Inverse Cavity Thickness $\frac{1}{d}$ ["+inv_um+"]")
      plt.xlim(xlim)
  else:
      fig.supxlabel("Cavity Thickness [um]")

  if return_fit:
    return fig, axs, om_os, cs
  return fig, axs

# %%
if __name__ == "__main__":
  
  domain = [1-0.5j, 2.5+0.05j]
  domain = [0.7-0.7j, 5.0+0.05j]

  hbar_omega = np.linspace(1, 2.5, 401)
  inv_d = np.linspace(1/1.2, 14, 1021)
  E, T = np.meshgrid(hbar_omega, 1/inv_d)

  # %%
  # %matplotlib inline
  fig, axs, om_os, cs = plot_splitting(1,3,1,1,domain, return_fit=True, plot_dots=False)
  plt.savefig("out/Fit_Hamiltonian.pdf")