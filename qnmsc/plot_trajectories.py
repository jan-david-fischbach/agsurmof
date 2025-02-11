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
# ---

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
from qnmsc.mpl_config import um, inv_um, mm
# %config InlineBackend.figure_format='retina'

if __name__=="__main__":
  import qnmsc.mpl_config
  qnmsc.mpl_config.config()
  DEBUG=False
  domain = [1-0.5j, 2.5+0.05j]

# %%
import numpy as np
import pickle
from qnmsc.materials import eps_surmof, eps_ag, surmof_material_data, to_eV, to_omega
from qnmsc.track_qnms import filename, track_qnms
import diffaaable

# %%
def calc_material_poles(scale_osc, scale_damping):
  """Material poles in $s^{-1}$

  Args:
      scale_osc (float): Scaling factor for oscillator strength 
      scale_damping (float): and damping

  Returns:
      complex: Material poles
  """

  mat = surmof_material_data(
    scale_osc=scale_osc, scale_damping=scale_damping
  )
  omega_0, gamma, intensity, eps_background = mat

  material_poles = np.sqrt(omega_0**2-1/4*gamma**2) - 1j*gamma/2 
  return material_poles

def load_data(npoles, scale_osc, scale_damping, domain):
  fname = filename(npoles, scale_osc, scale_damping, domain)
  with open(fname, "rb") as file:
      results = pickle.load(file)

  poles = results['poles']
  residues = results['residues']
  thickness = results['thickness']
  thickness=np.array(thickness)
  material_poles = to_eV(calc_material_poles(scale_osc, scale_damping))[:npoles]
  return poles, residues, thickness, material_poles

# %%
def to_THz(hbar_omega):
  return to_omega(hbar_omega)/(2*np.pi)/1e12

unit_conversion = {
  "eV": lambda x: x,
  "THz": to_THz,
}

unit_str = {
  "eV": "eV",
  "THz": "THz",
}

qty_str = {
  "eV": "\hbar \omega",
  "THz": "f",
}

# %%
def plot_thickness(npoles, scale_osc, scale_damping, domain, 
  unit = "eV", inv=True, horizontal=False
  ):

  poles, residues, thickness, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )

  for t, p, r in zip(thickness, poles, residues):
    if p.size:
      p = unit_conversion[unit](p)
      y = 1/t if inv else t
      y = [y]*len(p)
      x = np.real(p)
      if horizontal:
        x,y = y,x
      plt.scatter(x, y, color="k", marker=".", 
              alpha=np.clip(10*np.sqrt(np.abs(r)), 0, 1))

  xlabel = f"$\Re\{{{qty_str[unit]}\}}$ [{unit_str[unit]}]"
  if inv:
    ylabel = f"$1/d$ [{inv_um}]"
    plt.ylim(0, 8)
  else:
    ylabel = f"$d$ [{um}]"

  if horizontal:
    xlabel, ylabel = ylabel, xlabel
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

# %%
def plot_trajectories(npoles, scale_osc, scale_damping, domain, 
  plot_domain = [1-0.06j, 2.5+0.01j], fig_width = 90*mm,
  unit = "eV", axs=None, cbar=True, plot_neg_eps_r=False, d_limit=np.inf,
  ):

  c_ = unit_conversion[unit]
  u_ = unit_str[unit]
  q_ = qty_str[unit]

  poles, residues, thickness, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )
  poles_tracked, residues_tracked = track_qnms(poles, residues)

  if axs is None:
    fig, axs = plt.subplots(
      2, 1, sharex=True, 
      figsize=(fig_width,80*mm), constrained_layout=True
    )
  plt.sca(axs[0])

  # hbar_omega is the photon energy
  # as such it is denoted as e and E for brevity
  e_r = np.linspace(plot_domain[0].real, plot_domain[1].real, 1200)
  e_i = np.linspace(plot_domain[0].imag, plot_domain[1].imag, 400)
  E_r, E_i = np.meshgrid(e_r, e_i)
  E = E_r +1j*E_i

  eps_r = eps_surmof(e_r, npoles, scale_osc, scale_damping)
  lines = 0.5*(e_r[:-1]+e_r[1:])[np.diff(eps_r>0)!=0]
  
  eps = eps_surmof(E, npoles, scale_osc, scale_damping)
  ds = 20
  _, _, _, poles = diffaaable.aaa(c_(E[::ds, ::ds]), eps[::ds, ::ds])
  _, _, _, zeros = diffaaable.aaa(c_(E[::ds, ::ds]), 1/eps[::ds, ::ds])

  plt.pcolormesh(c_(E_r), c_(E_i), np.real(eps), norm=mpl.colors.SymLogNorm(linthresh=0.1,
                      vmin=-1000.0, vmax=1000.0, base=10), shading='gouraud',
                      cmap="RdBu", rasterized=True)

  if cbar:
    cb = plt.colorbar(label="$\Re\{\\varepsilon_\mathrm{r}\}$", ticks=[-1e3, -1, 0, 1, 1e3])
    cb.set_ticklabels(["-$10^3$", -1, 0, 1, "$10^3$"])

  plt.ylim(plt.ylim())
  plt.xlim(plt.xlim())

  ## Material Poles and Zeros
  plt.scatter(
    poles.real, poles.imag, 
    marker="x", color="k"
  )
  plt.scatter(
    zeros.real, zeros.imag, 
    facecolors='none', edgecolors="k", linewidths=1
  )

  # Interpolate thicknesses to make plot smoother
  interp_d = np.linspace(min(thickness), min(max(thickness), d_limit), (len(thickness)-1)*10+1) [::-1]
  cmap = mpl.cm.viridis_r
  norm = mpl.colors.Normalize(vmin=min(interp_d), vmax=max(interp_d))
  colors_interp = cmap(norm(interp_d))

  for pole, res in zip(poles_tracked.T, residues_tracked.T):
    real = np.interp(interp_d, thickness, pole.real)
    imag = np.interp(interp_d, thickness, pole.imag)
    s = np.interp(interp_d, thickness, np.abs(res))/(-imag)
    plt.scatter(c_(real), c_(imag), c=colors_interp, edgecolor='none', s=s, rasterized=True)

  if cbar:
    plt.ylabel(f"$\Im\{{{q_}\}}$ [{u_}]")

  plt.axhline(0, color="k", lw=0.4)
  # Negative Real Eps regions
  if plot_neg_eps_r:
    plt.fill_between(c_(e_r), 0, 1, hatch="\\\\\\", where=eps_r.real<0, color="none", edgecolor="k", transform=plt.gca().get_xaxis_transform(), lw=0.5)

    plt.sca(axs[1])
    plt.fill_between(c_(e_r), 0, 1, hatch="\\\\\\", where=eps_r.real<0, color="none", edgecolor="k", transform=plt.gca().get_xaxis_transform(), lw=0.5, zorder=5)
  else:
    plt.sca(axs[1])

  for pole, res in zip(poles_tracked.T, residues_tracked.T):
    real = np.interp(interp_d, thickness, pole.real)
    imag = np.interp(interp_d, thickness, pole.imag)
    s = np.interp(interp_d, thickness, np.abs(res))/(-imag)
    plt.scatter(c_(real), interp_d, c=colors_interp, edgecolor='none', s=s, rasterized=True)

  plt.ylim(0, max(interp_d))
  plt.xlim(min(c_(e_r)), max(c_(e_r)))

  if cbar:
    plt.ylabel(f"$d$ [{um}]")
    plt.xlabel(f"$\Re\{{{q_}\}}$ [{u_}]")


# %%
if __name__ == "__main__":
    plot_trajectories(3, 1, 1, domain, unit="THz", plot_neg_eps_r=True, fig_width=180*mm)
    plt.savefig("out/ThreePole.pdf", dpi=600)

    # %%
    plot_trajectories(1, 1, 1, domain, unit="THz", fig_width=90*mm)
    plt.savefig("out/SinglePole.pdf", dpi=600)

    # %%
    oscs = [1, 0.25, 0.1, 0.025, 0.01][0:-1]
    plot_domain1 = [1.5-0.14j, 2+0.01j]
    plot_domain2 = [1.6-0.14j, 1.85+0.01j]

    fig, axss = plt.subplots(
        2, len(oscs), sharex="col", sharey="row",
        figsize=(180*mm,80*mm), constrained_layout=True
        )
    for osc, axs in zip(oscs, axss.T):
        plot_trajectories(
            1, osc, 1, domain, 
            plot_domain=plot_domain1 if osc==1 else plot_domain2, 
            unit="THz", axs=axs, cbar=False, d_limit=0.4
        )
    axs[0].set_title(f"{osc:.3f}")

    axss[0,0].set_title("Scaled Oscillator Strength")
    fig.supxlabel("$\Re\{f\}$ [THz]")
    axss[0,0].set_ylabel("$\Im\{f\}$ [THz]")
    axss[1,0].set_ylabel(f"$d$ [{um}]")
    plt.savefig("out/OscReduction.pdf", dpi=600)

    # %%
    # plot_thickness(3, 1, 1, domain)
    # plot_thickness(3, 1, 1, domain, inv=False)

    # %%



