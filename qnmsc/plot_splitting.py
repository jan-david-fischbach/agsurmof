# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
from qnmsc.track_qnms import filename, track_qnms
from qnmsc.plot_trajectories import plot_thickness, load_data, select_modes
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
  plt.legend(fontsize=5, labelcolor=c_font, loc=legend_loc)

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
    return fig, axs, om_os, Cs, param_interp
  return fig, axs

# %%
if __name__ == "__main__":
  from qnmsc.surmof_cavity import ag_surmof_cavity_smat
  
  domain0 = [1-0.5j, 2.5+0.05j]
  domain = [0.7-0.7j, 5.0+0.05j]

  hbar_omega = np.linspace(1, 2.5, 401)
  inv_d = np.linspace(1/1.2, 14, 1021)
  E, T = np.meshgrid(hbar_omega, 1/inv_d)

  smat = ag_surmof_cavity_smat( 
    E, T
  )

  # %%
  fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(90*mm, 80*mm), height_ratios=[3.5,1.5])
  plt.sca(axs[0])

  from matplotlib.colors import ListedColormap
  a = 0.7
  my_cmap = plt.cm.viridis(np.arange(plt.cm.viridis.N))
  my_cmap[:,0:3] *= a 
  my_cmap = ListedColormap(my_cmap)

  plt.pcolormesh(1/T, E, np.abs(smat['in', 'out'])**2, zorder=-2, rasterized=True, shading='gouraud', cmap=my_cmap)
  cbar = plt.colorbar(label="Transmissivity")

  poles, residues, thickness_0pole, _ = load_data(
    0, 1, 1, domain0
  )
  poles_tracked_0pole, residues_tracked_0pole = track_qnms(poles, residues)
  plt.plot(1/thickness_0pole, poles_tracked_0pole[:, 0], ".", color="white", markersize=2)
  plt.plot([],[], ".", color="white", markersize=2, label="0 pole mode")

  plt.tick_params(which='both', color="white")
  cbar.ax.tick_params(which='both', color="white")

  for which, length, width in zip(['major', 'minor'], [3.5, 2], [0.5,0.5]):
    plt.tick_params(which=which, length=length, width=width)
    cbar.ax.tick_params(which=which, length=length, width=width)

  fig, axs, om_os, Cs, param = plot_splitting(1,3,1,1,domain, axs=axs,
    c_fundamental="white",
    c_higher="white",
    c_grid=(0.4, 0.4, 0.4),
    lw_higher=0.35,
    c_fit="white",
    c_font="white",
    c_mat=["C9", "C1", "C2"],
    force_legend=True,
    xlim=(min(inv_d), max(inv_d)),
    ylim=(min(hbar_omega), max(hbar_omega)),
    legend_loc="lower right",
    plot_dots=False,
    return_fit=True,
  )

  ## Corrected estimated 0 pole mode
  plt.sca(axs[0])
  poles, residues, thickness, material_poles = load_data(
    3, 1, 1, domain
  )
  iVii = Cs / (-1* material_poles)
  om_os = np.array(om_os)

  corr = np.sum(iVii, axis=-1)
  corr_om_os = om_os + corr
  
  plt.plot(param, corr_om_os, "-", color="white", label="corrected 'uncoupled' cavity mode", lw=0.6, zorder=7)
  plt.legend(fontsize=5, labelcolor="white", loc="lower right")

  for i, ax in enumerate(axs):
    letter = chr(ord("a")+i)
    ax.annotate(
          f" ({letter})",
          xy=(0, 1), xycoords='axes fraction',
          xytext=(+0.5, -0.5), textcoords='offset fontsize',
          fontsize='medium', verticalalignment='top', fontfamily='serif',
          bbox=dict(facecolor=(1,1,1,0.8), edgecolor='none', pad=2.0))

  plt.savefig("out/Fit_Hamilonian_BG.pdf", dpi=600)

  # %%
  # %matplotlib inline
  fig, axs, om_os, Cs, param = plot_splitting(1,3,1,1,domain, return_fit=True, plot_dots=False)
  plt.savefig("out/Fit_Hamiltonian.pdf")

  # %%
  poles, residues, thickness, material_poles = load_data(
    3, 1, 1, domain
  )
  iVii = Cs / (-1* material_poles)
  om_os = np.array(om_os)

  corr = np.sum(iVii, axis=-1)
  corr_om_os = om_os + corr
  
  plt.plot(param, corr_om_os, "-")
  plt.plot(1/thickness_0pole, poles_tracked_0pole[:,0], ".")
  plt.plot(param, om_os, "-")

  # %%
  plot_splitting(1,1,1,1,domain, plot_dots=False)
  plt.savefig("out/Single_pole_splitting.pdf")

  # %%
  ## Experimentation
  exit()

  # %%
  fig, axs, om_os1, Cs1 = plot_splitting(1,1,1,1,domain, color_rabi="C0", return_fit=True)
  _, _,     om_os2, Cs2 = plot_splitting(2,1,1,1,domain, color_rabi="C1", return_fit=True, axs=axs)
  _, _,     om_os3, Cs3 = plot_splitting(3,1,1,1,domain, color_rabi="C1", return_fit=True, axs=axs)
  _, _,     om_os4, Cs4 = plot_splitting(4,1,1,1,domain, color_rabi="C2", return_fit=True, axs=axs, xlim=(0.3, 12))

  # %%
  cs1, cs2, cs3, cs4 = [np.real(np.sqrt(Cs)) for Cs in [Cs1, Cs2, Cs3, Cs4]]
  
  plt.plot(om_os1, cs1)
  plt.plot(om_os2, cs2)
  plt.plot(om_os3, cs3)
  plt.plot(om_os4, cs4)


# %%
om_oss = []
css = []
axs = None
for i, mode in enumerate([1,2,4,5]):
  fig, axs, om_os, Cs = plot_splitting(mode,1,1,1,domain, 
    color_rabi=f"C{i}", return_fit=True, axs=axs,
    xlim=(0.2,7), inv_d=True)
  om_oss.append(om_os)
  css.append(np.real(np.sqrt(Cs)))

# %%
for om_os, cs in zip(om_oss, css):
  plt.plot(om_os, cs)

# %%
