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
from qnmsc.plot_trajectories import plot_thickness, load_data
from qnmsc import inverse_eigenproblem
from qnmsc.mpl_config import um, inv_um, mm
import numpy as np

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format='retina'
if __name__ == "__main__":
  import qnmsc.mpl_config
  qnmsc.mpl_config.config()

# %%
domain = [1-0.5j, 2.5+0.05j]
domain = [0.7-0.7j, 5.0+0.05j]

# %%
def label_tracked_qnms(poles_tracked, thickness):
  plt.plot(thickness, poles_tracked.real, ".-")
  for i, ptf in enumerate(poles_tracked.T):
      filter = ~np.isnan(ptf)
      if not np.any(filter):
          continue
      plt.annotate(f"p{i}", (thickness[filter][0],ptf[filter].real[0]), fontsize=5)
      plt.annotate(f"p{i}", (thickness[filter][-1],ptf[filter].real[-1]),fontsize=5)
  plt.xlabel(f"$d$ [{um}]")
  plt.ylabel("$\hbar \omega$ [eV]")

def select_modes(npoles, scale_osc, scale_damping, domain, modenumber=1, force=False):
  fname = filename(npoles, scale_osc, scale_damping, domain)
  select_fname = fname.parent/(fname.stem+f".select{modenumber}.npy")

  if select_fname.is_file() and not force:
    selection = np.load(select_fname)
    return selection, modenumber

  poles, residues, thickness, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )
  poles_tracked, residues_tracked = track_qnms(poles, residues)

  plot_thickness(npoles, scale_osc, scale_damping, domain, inv=False, horizontal=True)
  label_tracked_qnms(poles_tracked, thickness)
  plt.show(block=True)

  while True:
    print("Please select the modes as a comma separated list")
    user_selection = input()
    select = user_selection.split(",")
    if len(select) != npoles + 1:
      print("Please give npoles + 1 select")
      continue

    try:
      select = [int(m) for m in select]
    except:
      print("Please provide the integer mode numbers only")
      continue

    select = np.array(select)
    max_mode = poles_tracked.shape[1]
    if np.any(select>max_mode):
      print(f"Mode number too large: maximum is {max_mode}")
      continue
    
    np.save(select_fname, select)
    return select, modenumber


# %%
def plot_splitting(
  modenumber, npoles, scale_osc, scale_damping, domain, 
  inv_L=True, ylim=(1.2, 2.5), xlim=(1,12), color_rabi="r",
  axs=None, return_fit=False,
  c_os          = (0.8, 0.8, 0.8),
  c_grid        = (0.8, 0.8, 0.8),
  c_fundamental = "gray",
  c_higher      = (0, 0, 0, 0.2),
  c_fit        = "k",
  c_font        = "k",
  force_legend  = False,
  legend_loc    = 'best'
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

  def populate_legend():
    plt.plot([],[], color=c_fundamental, label="fundamental QNMs")
    plt.plot([],[], color=c_higher, linewidth=1, label="higher order QNMs")
    plt.plot([],[], color=c_fit, linestyle="none", marker=".", label="coupling fit")
    plt.plot([],[], "--", color=c_os, zorder=5, label="'uncoupled' cavity mode\n(from fit)")

  param = 1/thickness if inv_L else thickness
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

  param_interp = np.linspace(min(param), max(param), 300)
  for i, p in enumerate(param_interp):
      interp_d = 1/p if inv_L else p
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
      plt.axhline(mat_pole.real, color=f"C{i}")#, linestyle="--")

  for pole in poles_tracked.T:
    plt.plot(param, pole.real, color=c_higher, linewidth=1, zorder=-1)

  plt.axvline(rabi_param, color=c_grid, zorder=-1)
  plt.ylim(ylim)
  plt.sca(axs[1])
  plt.axvline(rabi_param, color=c_grid, zorder=-1)

  plt.ylabel("Coupling $c_i$ [eV]")
  Cs = np.array(Cs)
  cs = np.sqrt(np.real(Cs))
  for i,coupling in enumerate(cs.T):
      plt.plot(param_interp, coupling, ".-", label=f"$p_{i+1}=\complexqty{{{material_poles[i]:.3f}}}{{eV}}$")
  stud = 0.1
  x = [rabi_param-stud,rabi_param+stud]
  plt.plot(x, [f_rabi.real/2]*2, color="r")
  plt.ylim(0, 1.1*max(f_rabi.real/2, np.nanmax(cs.flatten())))

  plt.legend(fontsize=6, title="Material Resonances", loc="lower right", frameon=True)
  if inv_L:
      fig.supxlabel(r"Inverse Cavity Thickness $\frac{1}{d}$ ["+inv_um+"]")
      plt.xlim(xlim)
  else:
      fig.supxlabel("Cavity Thickness [um]")

  if return_fit:
    return fig, axs, om_os, cs
  return fig, axs

# %%
from qnmsc.surmof_cavity import ag_surmof_cavity_smat

hbar_omega = np.linspace(1, 2.5, 401)
inv_d = np.linspace(1, 14, 421)
E, T = np.meshgrid(hbar_omega, 1/inv_d)

smat = ag_surmof_cavity_smat( 
  E, T
)

# %%
fig, axs = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(90*mm, 90*mm), height_ratios=[3,1])
plt.sca(axs[0])
plt.pcolormesh(1/T, E, np.abs(smat['in', 'out'])**2, zorder=-2, rasterized=True)
plt.colorbar(label="Transmissivity")
plot_splitting(1,3,1,1,domain, axs=axs,
  c_fundamental="white",
  c_higher=(1,1,1,0.5),
  c_fit="white",
  c_font="white",
  force_legend=True,
  xlim=(min(inv_d), max(inv_d)),
  ylim=(min(hbar_omega), max(hbar_omega)),
  legend_loc="lower right"
)

plt.savefig("out/Fit_Hamilonian_BG.pdf", dpi=600)

# %%
# %matplotlib inline
plot_splitting(1,3,1,1,domain)
plt.savefig("out/Fit_Hamiltonian.pdf")


# %%
fig, axs, om_os1, cs1 = plot_splitting(1,1,1,1,domain, color_rabi="C0", return_fit=True)
_, _,     om_os2, cs2 = plot_splitting(2,1,1,1,domain, color_rabi="C1", return_fit=True, axs=axs)
_, _,     om_os3, cs3 = plot_splitting(4,1,1,1,domain, color_rabi="C2", return_fit=True, axs=axs, xlim=(0.3, 12))

# %%
plt.plot(om_os1, cs1)
plt.plot(om_os2, cs2)
plt.plot(om_os3, cs3)

# %%
