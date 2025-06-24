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
import scipy.io
from qnmsc.mpl_config import um, inv_um, mm

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

    plt.plot(1/r_outer, mode_freq, **kwargs)
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
  for pol in [1]:
    for l in [1,2]:
      for mode in range(1, 10):
        o = str(osc_strength).replace('.', 'p')
        if osc_strength == 1:
          o = ''
        suffix = f"onepole{o}_fixedthickness20nm"

        color = cmap(osc_strength-1e-3)
        plot_trajectory(pol, l, mode, suffix=suffix, color=color)
    
  plt.plot([], [], color=color, label=f"{osc_strength:.2f}")


plt.xlabel(rf'$1/r_\mathrm{{outer}}$ [{inv_um}]')
plt.ylabel(rf'$k_0$ [$2\pi$ {inv_um}]')

plt.xlim(2.7, 14)
plt.ylim(7.5, 10)

plt.legend(title="Oscillator Strength")

# %%
