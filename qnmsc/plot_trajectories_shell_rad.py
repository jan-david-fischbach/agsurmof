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
import matplotlib.pyplot as plt
import matplotlib as mpl
from qnmsc.mpl_config import um, inv_um, mm
# %matplotlib inline
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
import matplotlib.colors as mcolors
import colorsys

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
    param = results['param']
    radius, thickness_surmof = zip(*param)  # <-- Extract from param
    radius = np.array(radius)
    thickness_surmof = np.array(thickness_surmof)
    material_poles = to_eV(calc_material_poles(scale_osc, scale_damping))[:npoles]
    return poles, residues, radius, thickness_surmof, material_poles


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
def label_tracked_qnms(poles_tracked, radius):
  plt.plot(radius, poles_tracked.real, ".-")
  for i, ptf in enumerate(poles_tracked.T):
      filter = ~np.isnan(ptf)
      if not np.any(filter):
          continue
      plt.annotate(f"p{i}", (radius[filter][0],ptf[filter].real[0]), fontsize=5)
      plt.annotate(f"p{i}", (radius[filter][-1],ptf[filter].real[-1]),fontsize=5)
  plt.xlabel(f"$d$ [{um}]")
  plt.ylabel("$\hbar \omega$ [eV]")
  
def select_modes(npoles, scale_osc, scale_damping, domain, modenumber=1, force=False):
  fname = filename(npoles, scale_osc, scale_damping, domain)
  select_fname = fname.parent/(fname.stem+f".select{modenumber}.npy")

  if select_fname.is_file() and not force:
    selection = np.load(select_fname)
    return selection, modenumber

  poles, residues, radius, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )
  poles_tracked, residues_tracked = track_qnms(poles, residues)

  plt.figure("Select Modes")
  plot_radius(npoles, scale_osc, scale_damping, domain, inv=False, horizontal=True)
  label_tracked_qnms(poles_tracked, radius)
  plt.title(f"Select Modes m={modenumber}")
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

    select = np.array(select, dtype=int)
    max_mode = poles_tracked.shape[1]
    if np.any(select>max_mode):
      print(f"Mode number too large: maximum is {max_mode}")
      continue
    
    select = select[select>=0] # enter -1 to avoid selecting a mode (e.g. if outside of domain)
    np.save(select_fname, select)
    return select, modenumber

# %%
def plot_radius(npoles, scale_osc, scale_damping, domain, 
  unit = "eV", inv=True, horizontal=False, use_alpha=False, resize=False
  ):

  poles, residues, radius, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )

  for t, p, r in zip(radius, poles, residues):
    if p.size:
      p = unit_conversion[unit](p)
      y = 1/t if inv else t
      y = [y]*len(p)
      x = np.real(p)
      if horizontal:
        x,y = y,x
      alpha = np.clip(10*np.sqrt(np.abs(r)), 0, 1) if use_alpha else 1
      size = np.abs(r)/np.abs(p.imag)**2 * 1e-6
      plt.scatter(x, y, color="k", marker=".", 
              alpha=alpha, s=size)

  xlabel = f"$\Re\{{{qty_str[unit]}\}}$ [{unit_str[unit]}]"
  if inv:
    ylabel = f"$1/r$ [{inv_um}]"
    plt.ylim(0, 8)
  else:
    ylabel = f"$r$ [{um}]"

  if horizontal:
    xlabel, ylabel = ylabel, xlabel
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)

# %%
def plot_trajectories(npoles, scale_osc, scale_damping, domain, 
  plot_domain = [1-0.06j, 2.5+0.01j], fig_width = 90*mm, fig_height = 80*mm,
  unit = "eV", axs=None, cbar=True, labels=True, plot_neg_eps_r=False, d_limit=np.inf, 
  color_radius=False, num_modes = 4, upsample=10, label_suffix=""
  ):

  c_ = unit_conversion[unit]
  u_ = unit_str[unit]
  q_ = qty_str[unit]

  poles, residues, radius, material_poles = load_data(
    npoles, scale_osc, scale_damping, domain
  )
  poles_tracked, residues_tracked = track_qnms(poles, residues)

  if axs is None:
    fig, axs = plt.subplots(
      2, 1, sharex=True, 
      figsize=(fig_width,fig_height), constrained_layout=True
    )
  else:
    fig = plt.gcf()
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

  pcm = plt.pcolormesh(c_(E_r), c_(E_i), np.real(eps), norm=mpl.colors.SymLogNorm(linthresh=0.1,
                      vmin=-1000.0, vmax=1000.0, base=10), shading='gouraud',
                      cmap="RdBu", rasterized=True)

  if cbar:
    cb = plt.colorbar(
      label="$\Re\{\\varepsilon_\mathrm{r}\}$", 
      ticks=[-1e3, -1, 0, 1, 1e3]
    )
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
    facecolors='none', edgecolors="k", linewidths=2
  )

  # Interpolate radiuses to make plot smoother
  interp_d = np.linspace(min(radius), min(max(radius), d_limit), (len(radius)-1)*upsample+1) [::-1]
  cmap = mpl.cm.viridis_r
  norm = mpl.colors.Normalize(vmin=min(interp_d), vmax=max(interp_d))
  colors_interp = cmap(norm(interp_d))

  for i, (pole, res) in enumerate(zip(poles_tracked.T, residues_tracked.T)):
    real = np.interp(interp_d, radius, pole.real)
    imag = np.interp(interp_d, radius, pole.imag)
    s = np.interp(interp_d, radius, np.abs(res))/(-imag)
    plt.scatter(c_(real), c_(imag), c=colors_interp, edgecolor='none', s=s, rasterized=True)

  if labels:
    plt.ylabel(f"$\Im\{{{q_}\}}$ [{u_}]")

  plt.axhline(0, color="k", lw=0.4)
  # Negative Real Eps regions
  if plot_neg_eps_r:
    plt.fill_between(c_(e_r), 0, 1, hatch="\\\\\\", where=eps_r.real<0, color="none", edgecolor="k", transform=plt.gca().get_xaxis_transform(), lw=0.5)

    plt.sca(axs[1])
    plt.fill_between(c_(e_r), 0, 1, hatch="\\\\\\", where=eps_r.real<0, color="none", edgecolor="k", transform=plt.gca().get_xaxis_transform(), lw=0.5, zorder=5)
  else:
    plt.sca(axs[1])

  all_modes = set(range(len(poles_tracked.T)))
  for i in range(1, num_modes+1):
    selection, _ = select_modes(npoles, scale_osc, scale_damping, domain, modenumber=i)
    all_modes -= set(selection)

    for j, (pole, res) in enumerate(zip(poles_tracked.T[selection], residues_tracked.T[selection])):
      real = np.interp(interp_d, radius, pole.real)
      imag = np.interp(interp_d, radius, pole.imag)
      s = np.interp(interp_d, radius, np.abs(res))/(-imag)

      c0_rgba = mcolors.to_rgba(f"C{i-1}")
      c0_hsv = colorsys.rgb_to_hsv(*c0_rgba[:3])
      new_saturation = c0_hsv[1] * ((npoles+1)-j)/(npoles+1)
      new_rgb = colorsys.hsv_to_rgb(c0_hsv[0], new_saturation, c0_hsv[2])
      new_rgba = (*new_rgb, c0_rgba[3])

      c = colors_interp if color_radius else new_rgba

      plt.scatter(c_(real), interp_d, c=c, edgecolor='none', s=s, rasterized=True)
  
  all_modes = np.array(list(all_modes), dtype=int)
  for pole, res in zip(poles_tracked.T[all_modes], residues_tracked.T[all_modes]):
      real = np.interp(interp_d, radius, pole.real)
      imag = np.interp(interp_d, radius, pole.imag)
      s = np.interp(interp_d, radius, np.abs(res))/(-imag)
      plt.scatter(c_(real), interp_d, c=[0.8,0.8,0.8], edgecolor='none', s=s, rasterized=True)

  plt.ylim(0, max(interp_d))
  plt.xlim(min(c_(e_r)), max(c_(e_r)))
  fig.align_ylabels()

  for i, ax in enumerate(axs):
    letter = chr(ord("a")+i)
    ax.annotate(
          f" ({letter}{label_suffix})",
          xy=(0, 1), xycoords='axes fraction',
          xytext=(+0.5, -0.5), textcoords='offset fontsize',
          fontsize='medium', verticalalignment='top', fontfamily='serif',
          bbox=dict(facecolor=(1,1,1,0.8), edgecolor='none', pad=2.0))


  if labels:
    plt.ylabel(f"$d$ [{um}]")
    plt.xlabel(f"$\Re\{{{q_}\}}$ [{u_}]")

  return pcm, cmap, norm

def plot_thickness(npoles, scale_osc, scale_damping, domain, 
              unit="eV", inv=True, horizontal=False, use_alpha=True):
    poles, residues, radius, thickness_surmof, material_poles = load_data(
        npoles, scale_osc, scale_damping, domain
    )

    for b, p, r in zip(thickness_surmof, poles, residues):
        if p.size:
            p = unit_conversion[unit](p)
            y = 1/b*1e3 if inv else b*1e3
            y = [y] * len(p) 
            x = np.real(p)
            #pdb.set_trace()

            if horizontal:
                x, y = y, x
            alpha = np.clip(0.1*np.abs(1/r)**0.05, 0, 1) if use_alpha else 1
            size = 5
            plt.scatter(x, y, color="r", marker=".", alpha=alpha, s=size)

    xlabel = f"$\\Re\\{{{qty_str[unit]}\\}}$ [{unit_str[unit]}]"
    ylabel = f" Inverse Thickness ($1/nm$)" if inv else f"Thickness (nm)"

    if horizontal:
        xlabel, ylabel = ylabel, xlabel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# %%
if __name__ == "__main__":
    plt.figure()
    plot_thickness(1, 1, 1, domain, inv=False, horizontal = True)
    plt.xlim(0, 200)
    plt.ylim(1.65, 1.9)
    plt.savefig("out/Elli_1pole_rad_shell_kz0.pdf", dpi=1200)
    plt.show()

# %%

# %%
