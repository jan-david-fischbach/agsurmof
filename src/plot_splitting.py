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
from track_qnms import filename, track_qnms
from plot_trajectories import plot_thickness, load_data
import numpy as np

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format='retina'
if __name__ == "__main__":
  from mpl_config import um, inv_um

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
  label_tracked_qnms(poles_tracked, residues_tracked, thickness)
  plt.show(block=False)

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
DEBUG=True
if DEBUG:
  npoles = 3
  scale_osc = 1
  scale_damping = 1
  domain = [1-0.5j, 2.5+0.05j]

  poles, residues, thickness, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )
  poles_tracked, residues_tracked = track_qnms(poles, residues)

  selection, modenumber = select_modes(npoles, scale_osc, scale_damping, domain)
  selected = poles_tracked[:, selection]

  plt.plot(thickness, selected)

  splitting = np.abs(np.real(selected[:, 0]-selected[:, -1]))

  if DEBUG:
    plt.twinx()
    plt.plot(thickness, splitting, "r--")
    plt.ylabel("Splitting")
    
  try:
    rabi_idx = np.nanargmin(splitting)
    f_rabi = np.nanmin(splitting)
    rabi_ok = True
  except ValueError as e:
    rabi_ok = False

# %%
INV_L = True
param = 1/thickness if INV_L else thickness

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4, 4), height_ratios=[3,1])
plt.sca(axs[0])

f_mode = selected.real
plt.plot(param, f_mode, color="gray")
plt.plot([],[], color="gray", label="QNMs")

import inverse_eigenproblem
Cs=[]
om_os=[]
evs_fit = []
for i, t in enumerate(thickness):
    evs = selected[i]
    om_o, *couplings = inverse_eigenproblem.solve_inv_eig(
        evs, material_poles
    )
    Cs.append(couplings)
    om_os.append(om_o)
    evs_fit.append(inverse_eigenproblem.test_fwd_eig(om_o, material_poles, np.sqrt(couplings)))

plt.plot(param, np.array(evs_fit).real, "k.")
plt.plot([], [], "k.", label="coupling fit")
plt.plot(param, np.array(om_os), "--", color=(0.8, 0.8, 0.8), zorder=5, label="'uncoupled' cavity mode\n(from fit)")
plt.vlines([param[rabi_idx]],*f_mode[rabi_idx, [0, -1]], color="r", label=f"traditional\n$\Omega_\mathrm{{Rabi}}=2\cdot{f_rabi.real/2:.3f}[\mathrm{{eV}}]$")
plt.ylabel("$\hbar \omega$ [eV]")
plt.legend(fontsize=6)

for i,mat_pole in enumerate(material_poles):
    plt.axhline(mat_pole.real, color=f"C{i}", linestyle="--")

plt.sca(axs[1])

plt.ylabel("Coupling [eV]")
Cs = np.array(Cs)
cs = np.sqrt(np.real(Cs))
for i,coupling in enumerate(cs.T):
    plt.plot(param, coupling, "--", label=f"$p_{i+1}={material_poles[i]:.3f}$ eV")
plt.axhline(f_rabi.real/2, color="r")
plt.ylim(0, 1.1*f_rabi.real/2)

plt.legend(fontsize=6, title="Material Resonances", loc="lower right")
if INV_L:
    fig.supxlabel("Inverse Cavity Thickness [1/um]")
    plt.xlim(2,12)
else:
    fig.supxlabel("Cavity Thickness [um]")
