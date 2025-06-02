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
from qnmsc.surmof_cavity import ag_surmof_cavity_smat
from qnmsc.plot_trajectories import calc_material_poles, to_eV, unit_conversion, load_data, track_qnms
from qnmsc.plot_splitting import plot_splitting
import numpy as np
import matplotlib.pyplot as plt
import diffaaable


# %%
from qnmsc.mpl_config import um, inv_um, mm
import qnmsc.mpl_config
qnmsc.mpl_config.config()

# %%
hbar_omega = np.linspace(1.2, 2.3, 201)
resonant_cavity_thickness = 1/4.780315
domain = [1-0.5j, 2.5+0.05j]


# %%
s_oscs = [0.025, 0.05, 0.1, 1]
f_rabis = []
gs = []
avg_loss = 0.072 # eV
for i, scale_osc in enumerate(s_oscs):
  fig, axs, om_os, Cs, param_interp, rabi_param, f_rabi = plot_splitting(
    1, 1, scale_osc, 1, domain=domain, n_interp=int(50),
    color_rabi = "r" if i>=1 else "none", inv_d=False, return_rabi=True
  )
  f_rabis.append(f_rabi)
  gs.append(np.sqrt(np.abs(Cs[np.argmin(np.abs(param_interp - resonant_cavity_thickness))][0])))



# %%
fig, axss = plt.subplots(2, 4, figsize=(180*mm, 80*mm), sharey="row", sharex=True, constrained_layout=True)
c_ = unit_conversion['THz']

axs = axss[0]
for i, scale_osc in enumerate(s_oscs):
  plt.sca(axs[i])

  axs[i].set_title(f"{scale_osc:.3f}")

  scale_damping = 1
  smat = ag_surmof_cavity_smat(
      hbar_omega, resonant_cavity_thickness, 1, 
      scale_osc, scale_damping, 
    )

  mat_pole = to_eV(calc_material_poles(scale_osc, scale_damping))[0]
  
  t = smat['in', 'out']

  Tran = np.abs(smat['in', 'out'])**2
  Refl = np.abs(smat['in', 'in'])**2
  Abs = 1 - Tran - Refl

  plt.plot(c_(hbar_omega), Tran, lw=0.25, color="k", label="transmission")
  plt.plot(c_(hbar_omega), Refl, "--k", lw=0.4, label="reflection")
  plt.plot(c_(hbar_omega), Abs, "k", label="absorption")
  plt.xlim(min(c_(hbar_omega)), max(c_(hbar_omega)))

  fit = diffaaable.aaa(hbar_omega, t)
  poles = fit[3]

  pole_mask = np.logical_and(poles.real > 1.3, poles.real < 2.3)

  residues = diffaaable.core.residues(*fit)
  
  poles = poles[pole_mask]
  residues = residues[pole_mask]
  
  sorter = np.argsort(-np.abs(poles-mat_pole))
  poles = poles[sorter][:2]
  residues = residues[sorter][:2]

  contributions = residues[None, :] / (hbar_omega[:, None] - poles[None, :])
  for j, contrib in enumerate(contributions.T):
    #plt.plot(hbar_omega, np.real(contrib), linestyle='--', color=f'C{j}')
    #plt.plot(hbar_omega, np.abs(contrib), color='gray')

    plt.plot(c_(hbar_omega), np.abs(contrib)**2, color=f'C{j}', alpha=0.4, label=f"mode {j+1}")
  #plt.plot(hbar_omega, np.real(np.sum(contributions, axis=-1)), color=f'C{i}')

  print(f"poles: {poles}; residues: {residues}")
  # for pole in poles:
  #   plt.axvline(pole, color = f'C{i}')

  plt.text(390, 1, f"$\\frac{{\Omega_{{\mathrm{{Rabi}}}}}}{{2}} \\approx {f_rabis[i]/2:.3f} \;$ eV\n$|g| \\approx {gs[i]:.3f} \;$ eV\n$\gamma_\mathrm{{avg}} \\approx {avg_loss} \;$ eV", size=4, ha="right")

axs[0].set_title(f"Oscillator Strength Scaling:\n{s_oscs[0]:.3f}")
axs[0].set_ylabel("Observable")
axs[-1].legend(loc='upper right', fontsize=5)
plt.ylim((0, None))

axs = axss[1]
thicknesses = np.linspace(0.0, 0.41, 201)
lw_qnm=0.2
for i, scale_osc in enumerate([0.025, 0.05, 0.1, 1]):

  
  plt.sca(axs[i])
  HO, T = np.meshgrid(hbar_omega, thicknesses)
  smat = ag_surmof_cavity_smat(
      HO, T, 1, 
      scale_osc, scale_damping, 
    )
  
  Tran = np.abs(smat['in', 'out'])**2
  Refl = np.abs(smat['in', 'in'])**2

  cm = plt.pcolormesh(c_(HO), T, Tran, vmin=0, vmax=0.4, rasterized=True)
  plt.axhline(resonant_cavity_thickness, color='white', linestyle='--', label="$\delta = 0$")


  poles, residues, thickness, material_poles = load_data(
    1, scale_osc, scale_damping, domain
  )
  poles_tracked, residues_tracked = track_qnms(poles, residues)
  plt.plot(c_(poles_tracked.real), thickness, color="white", lw=lw_qnm)

  plt.tick_params(which='both', color="white")


plt.plot([],[], color='white', lw=lw_qnm, label="QNMs")
plt.ylim(min(thicknesses), max(thicknesses))
axs[0].set_ylabel("$d$ [um]")
axs[-1].legend(labelcolor='white')

cbar = plt.colorbar(cm, ax=axs, label="$T = |t|^2$")
cbar.ax.tick_params(which='both', color="white")

fig.supxlabel(r"$\Re\{ f \}$ [THz]")
plt.savefig("out/OscReductionObservable.pdf", bbox_inches='tight')

# %%
