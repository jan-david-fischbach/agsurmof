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

# %% [markdown]
# #
# ## TODO 
# - ask why shift of 20e-3 nm? -> the saved radius is the core radius
# - ask for residues
# - ask for region at larger radii

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.io
from qnmsc.mpl_config import um, inv_um, mm
from qnmsc.materials import eps_surmof

import diffaaable

import qnmsc.mpl_config
qnmsc.mpl_config.config()

# %%
pols = ['TE', 'TM']
colors = ["C9", "C1"]
l_strings = ['', 'dipole', 'quadrupole'] 
def load_data(pol=0, l=1, mode=1, prefix="SURMOF_SILVER", suffix='fixedthickness20nm'):
  pol_str = pols[pol]
  l_str = l_strings[l]

  folder = "assets/in/3poles_surmof_20nmAu/"
  return scipy.io.loadmat(f"{folder}/{prefix}_{pol_str}_particle_{l_str}_mode{mode}_{suffix}.mat", struct_as_record=False)


# %%
def plot_trajectory(pol=0, l=1, mode=1, prefix="SURMOF_SILVER", suffix='fixedthickness20nm', **kwargs):
  try:
    data = load_data(pol, l, mode, prefix, suffix)
    full_tm = data['full_tm']

    full_tm.shape
    mode_freq = full_tm[:, 0]
    r_outer = full_tm[:, 1] + 20e-3 # TODO ask why this shift?!

    plt.plot(1/r_outer, mode_freq.real, **kwargs)
  except FileNotFoundError as e:
    return


# %%
background_data = scipy.io.loadmat("assets/in/3poles_surmof_20nmAu/Data_Material1SURMOF_Material2SILVERSCS_ABS_maps_SURMOF_SILVER_upquadrupole_fixedlayer20nm_allpoles.mat", struct_as_record=False)

# %%
from matplotlib.colors import ListedColormap
a = 0.6
my_cmap = plt.cm.viridis(np.arange(plt.cm.viridis.N))
my_cmap[:,0:3] *= a 
my_cmap = ListedColormap(my_cmap)

# %%
radius_outer = background_data['radiusf']
k0 = 2*np.pi/background_data['lam']

R, K = np.meshgrid(radius_outer[1], k0)
plt.pcolormesh(1/R/1e6, K/1e6, background_data['exblk'], zorder=-2, rasterized=True, shading='gouraud', cmap=my_cmap)
cbar = plt.colorbar(label="Extinction XSection")
cbar.ax.tick_params(which='both', color="white")

for pol in [0,1]:
  for l in [1,2]:
    for mode in range(1, 10):
      plot_trajectory(pol, l, mode, color=colors[pol], lw=0.6, ls='-' if l == 1 else '--')

plt.plot([], [], color=colors[0], label=pols[0])
plt.plot([], [], color=colors[1], label=pols[1])
plt.plot([], [], color="white", ls="-",  label="dipole")
plt.plot([], [], color="white", ls="--", label="quadrupole")

plt.legend(labelcolor="white", loc="lower right")
plt.xlabel(rf'$1/r_\mathrm{{outer}}$ [{inv_um}]')
plt.ylabel(rf'$k_0$ [$2\pi$ {inv_um}]')

plt.xlim(3, 14)
plt.ylim(7, 11)
plt.tick_params(which='both', color="white")

# %%
cmap = plt.cm.viridis
for osc_strength in [0.1, 0.25, 0.5, 1]:
  o = str(osc_strength).replace('.', 'p')
  if osc_strength == 1:
    o = ''
  suffix = f"onepole{o}_fixedthickness20nm"
  color = cmap(osc_strength-1e-3)
  for pol in [1]:
    for l in [1,2]:
      for mode in range(1, 10):

        plot_trajectory(pol, l, mode, suffix=suffix, color=color)
    
  plt.plot([], [], color=color, label=f"{osc_strength:.2f}")


plt.xlabel(rf'$1/r_\mathrm{{outer}}$ [{inv_um}]')
plt.ylabel(rf'$k_0$ [$2\pi$ {inv_um}]')

plt.xlim(2.7, 14)
plt.ylim(7.5, 10)

plt.legend(title="Oscillator Strength")


# %%
def plot_cplx_trajectory(pol=0, l=1, mode=1, prefix="SURMOF_SILVER", suffix='fixedthickness20nm', **kwargs):
  try:
    data = load_data(pol, l, mode, prefix, suffix)
    full_tm = data['full_tm']

    full_tm.shape
    mode_freq = full_tm[:, 0]
    r_outer = full_tm[:, 1] + 20e-3 # TODO ask why this shift?!

    plt.plot(mode_freq.real, mode_freq.imag, **kwargs)
  except FileNotFoundError as e:
    return


# %%
r_outer_lims = [0.07, 0.37]

from scipy.constants import h, c as c0,e
k0_to_eV = 1e6*h*c0/e/(2*np.pi)

def plot_single_gradient_trajectory(pol=0, l=1, mode=1, 
  prefix="SURMOF_SILVER", suffix='fixedthickness20nm', 
  interp_d=np.linspace(*r_outer_lims, 201), colors="k", **kwargs):

  try:
    data = load_data(pol, l, mode, prefix, suffix)
  except FileNotFoundError as e:
    return

  full_tm = data['full_tm'][::-1]
  pole = full_tm[:, 0] * k0_to_eV
  r_outer = np.real(full_tm[:, 1] + 20e-3)
  
  # print("r_outer: ", max(r_outer), min(r_outer))
  # print("interp_d: ", max(interp_d), min(interp_d))

  res = full_tm[:, 3] * k0_to_eV

  #plt.plot(pole.real, pole.imag, **kwargs)

  nans = {'left': np.nan, 'right': np.nan}
  real = np.interp(interp_d, r_outer, pole.real, **nans)
  imag = np.interp(interp_d, r_outer, pole.imag, **nans)
  s = np.interp(interp_d, r_outer, np.abs(res))/(-imag) * 100
  plt.scatter(real, imag, 
    c=colors, edgecolor='none', 
    #s=s, 
    rasterized=True)


def plot_real_trajectory(pol=0, l=1, mode=1, prefix="SURMOF_SILVER", suffix='fixedthickness20nm', **kwargs):
  try:
    data = load_data(pol, l, mode, prefix, suffix)
  except FileNotFoundError as e:
    return

  full_tm = data['full_tm']

  full_tm.shape
  mode_freq = full_tm[:, 0] * k0_to_eV
  r_outer = full_tm[:, 1] + 20e-3 # TODO ask why this shift?!

  plt.plot(mode_freq.real, r_outer, **kwargs)

def planar_analogous(osc_strength, axs, plot_domain = [1.25-0.6j, 2.25+0.05j], colors=[]):
  """
    Plot analogous to the plot_trajectories file for the planar cavities
  """

  plt.sca(axs[0])
  o = str(osc_strength).replace('.', 'p')
  if osc_strength == 1:
    o = ''
  suffix = f"onepole{o}_fixedthickness20nm"
  
  e_r = np.linspace(plot_domain[0].real, plot_domain[1].real, 1200)
  e_i = np.linspace(plot_domain[0].imag, plot_domain[1].imag, 400)
  E_r, E_i = np.meshgrid(e_r, e_i)
  E = E_r +1j*E_i

  eps = eps_surmof(E, 1, osc_strength, 1)
  ds = 20
  _, _, _, poles = diffaaable.aaa(E[::ds, ::ds], eps[::ds, ::ds])
  _, _, _, zeros = diffaaable.aaa(E[::ds, ::ds], 1/eps[::ds, ::ds])


  plt.xlim(plot_domain[0].real, plot_domain[1].real)
  plt.ylim(plot_domain[0].imag, plot_domain[1].imag)
  
  interp_d = np.linspace(*r_outer_lims, 4001)

  cmap = plt.cm.viridis_r
  thickness_color_norm = mpl.colors.Normalize(vmin=min(interp_d), vmax=max(interp_d))
  colors_interp = cmap(thickness_color_norm(interp_d))

  for pol in [1]:
    for l in [1,2]:
      for mode in range(1, 10):
        plot_single_gradient_trajectory(pol, l, mode, suffix=suffix, 
          interp_d=interp_d, colors=colors_interp)

  ## Material Poles and Zeros
  plt.scatter(
    poles.real, poles.imag, 
    marker="x", color="k"
  )
  plt.scatter(
    zeros.real, zeros.imag, 
    facecolors='none', edgecolors="k", linewidths=1
  )

  plt.sca(axs[1])

  for pol in [1]:
    for l in [1,2]:
      for mode in range(1, 10)[::-1]:
        if mode <= len(colors):
          color = colors[mode-1]
        else:
          color = 'gray'
        plot_real_trajectory(pol, l, mode, suffix=suffix, color=color)


# %%
mode_to_color = {
  1:   ['C1', 'C2', 'C0', 'C0'],
  0.5: ['C0', 'C2', 'C0'],
  0.25: ['gray', 'C0', 'C0', 'C1', 'C2'],
  0.1: ['C0', 'C0', 'gray', 'C1']
}

# %%
fig, axs = plt.subplots(2, 4, sharey="row", figsize=(180*mm,70*mm), sharex='col')

for i, osc_strength in enumerate([0.1, 0.25, 0.5, 1]):
  planar_analogous(osc_strength, axs = axs[:, i], colors = mode_to_color[osc_strength])

axs[0, 0].set_ylabel("$\Im\{\hbar \omega\}$ [eV]")
axs[1, 0].set_ylabel(rf'$r_\mathrm{{outer}}$ [{inv_um}]')

fig.supxlabel(r'$\Re\{\hbar \omega\}$ [eV]')
fig.align_ylabels()

axs[1, 0].set_ylim(0.07, 0.37)

# %%

# %%
