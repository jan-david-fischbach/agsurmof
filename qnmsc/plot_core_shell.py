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
# - Why does orange line stop?
# - Why do the modes not run into zero?

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

import scipy.io
from qnmsc.mpl_config import um, inv_um, mm
from qnmsc.materials import eps_surmof, to_eV
from qnmsc.plot_trajectories import calc_material_poles

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
    s=2,
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

def planar_analogous(osc_strength, axs, plot_domain = [1.25-0.6j, 2.25+0.05j], colors=[], label_suffix=''):
  """
    Plot analogous to the plot_trajectories file for the planar cavities
  """

  o = str(osc_strength).replace('.', 'p')
  if osc_strength == 1:
    o = ''
  suffix = f"onepole{o}_fixedthickness20nm"
  
  e_r = np.linspace(plot_domain[0][0].real, plot_domain[0][1].real, 41)
  e_i = np.linspace(plot_domain[0][0].imag, plot_domain[0][1].imag, 61)
  E_r, E_i = np.meshgrid(e_r, e_i)
  E = E_r +1j*E_i

  eps = eps_surmof(E, 1, osc_strength**2, 1)
  ds = 1
  _, _, _, poles = diffaaable.aaa(E[::ds, ::ds], eps[::ds, ::ds])
  _, _, _, zeros = diffaaable.aaa(E[::ds, ::ds], 1/eps[::ds, ::ds])

  
  interp_d = np.linspace(*r_outer_lims, 2001)

  cmap = plt.cm.viridis_r
  thickness_color_norm = mpl.colors.Normalize(vmin=min(interp_d), vmax=max(interp_d))
  colors_interp = cmap(thickness_color_norm(interp_d))

  for i in [1, 0]:
    plt.sca(axs[i])
    plt.xlim(plot_domain[i][0].real, plot_domain[i][1].real)
    plt.ylim(plot_domain[i][0].imag, plot_domain[i][1].imag)

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

  plt.sca(axs[2])

  for pol in [1]:
    for l in [1,2]:
      for mode in range(1, 10)[::-1]:
        if mode <= len(colors):
          color = colors[mode-1]
        else:
          color = 'gray'
        plot_real_trajectory(pol, l, mode, suffix=suffix, color=color)


  for i, ax in enumerate(axs):
    if i==1:
      continue
    letter = chr(ord("a")+i//2)
    ax.annotate(
          f" ({letter}{label_suffix})",
          xy=(0, 1), xycoords='axes fraction',
          xytext=(+0.5, -0.5), textcoords='offset fontsize',
          fontsize='medium', verticalalignment='top', fontfamily='serif',
          bbox=dict(facecolor=(1,1,1,0.8), edgecolor='none', pad=2.0))

  axs[0].spines[['bottom']].set_visible(False)
  axs[0].get_xaxis().tick_top()

  axs[1].spines[['top']].set_visible(False)
  axs[1].get_xaxis().tick_bottom()

  mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=thickness_color_norm)
  return mappable

# %%
mode_to_color = { # 'C2' is the passing (weakly coupled) mode
  1:   ['C1', 'C3', 'C0', 'C0', 'C2'],
  0.5: ['C0', 'C1', 'C0', 'C2'],
  0.25: ['C2', 'C0', 'C0', 'C1', 'C3'],
  0.1: ['C0', 'C0', 'C2', 'C1']
}

# %%
background_data['poles']*k0_to_eV

# %%
to_eV(calc_material_poles(1, 1))

# %%
fig, axs = plt.subplots(3, 5, sharey="row", figsize=(180*mm,70*mm), sharex='col', height_ratios=[1.5, 0.5, 1], width_ratios=[1]*4+[0.15]) #constrained_layout=True,


plot_domain2 = [[1.6-0.1199j, 1.85+0.02j], [1.6-0.53j, 1.85-0.4801j]]

plot_domain1 = [[1.3-0.1199j, 2.2 +0.02j], [1.3-0.53j, 2.2-0.4801j]]

for i, osc_strength in enumerate([0.1, 0.25, 0.5, 1]):
  plot_domain = plot_domain1 if osc_strength == 1 else plot_domain2
  mappable = planar_analogous(
    osc_strength, axs = axs[:, i], 
    colors = mode_to_color[osc_strength], label_suffix=f"{i+1}",
    plot_domain=plot_domain
  )
  axs[0, i].set_title(f"{osc_strength**2:.3f}")

cbar_ax = axs[0, -1]
cbar = plt.colorbar(ax=cbar_ax, mappable=mappable, fraction=1, label=rf'$r_\mathrm{{outer}}$ [{um}]')

axs[0, 0].set_title(f"$\eta$: {list(mode_to_color.keys())[-1]**2:.2f}")

axs[0, 0].set_ylabel("$\Im\{\hbar \omega\}$ [eV]")
axs[0, 0].yaxis.label.set_position((-0.2, 0.25))

axs[2, 0].set_ylabel(rf'$r_\mathrm{{outer}}$ [{um}]')

fig.supxlabel(r'$\Re\{\hbar \omega\}$ [eV]')
fig.align_ylabels()

axs[2, 0].set_ylim(0.07, 0.37)

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,
              linestyle="none", color='k', mec='k', mew=0.5, clip_on=False)

axs[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(base=0.05))  # y-axis ticks at multiples of 0.05
axs[1, 0].yaxis.set_minor_locator(ticker.MultipleLocator(base=0.01))

plt.tight_layout()
fig.subplots_adjust(hspace=0.08)

pos = cbar.ax.get_position()  # get current position
new_pos = [pos.x0, pos.y0 - 0.05, pos.width, pos.height]  # y0 shifted up
cbar.ax.set_position(new_pos)

for ax in axs.flatten():
  pos = ax.get_position()
  ax.set_position(pos)

for ax in axs[:, -1]:
  ax.axis('off')

for ax in axs[1]:
  pos = ax.get_position()
  ax.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

for axs in axs.T[:-1]:
  axs[0].plot([0, 1], [0, 0], transform=axs[0].transAxes, **kwargs)
  axs[1].plot([0, 1], [1, 1], transform=axs[1].transAxes, **kwargs)


plt.savefig('out/CoreShell.pdf', dpi=1200, bbox_inches='tight')
