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
#     display_name: qnmsc (3.11.10)
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
from matplotlib.patches import ConnectionPatch
import copy

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
  "eV": r"\hbar \tilde \omega",
  "THz": "f",
}

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
  plt.ylabel(r"$\hbar \tilde \omega$ [eV]")
  
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

  plt.figure("Select Modes")
  plot_thickness(npoles, scale_osc, scale_damping, domain, inv=False, horizontal=True)
  label_tracked_qnms(poles_tracked, thickness)
  plt.title(f"Select Modes m={modenumber}")
  plt.show(block=False)

  while True:
    print(f"Please select the modes (m={modenumber}) as a comma separated list")
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
    plt.close("Select Modes")
    return select, modenumber

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
def size(res, imag):
  s = res/(-imag)
  s = np.sqrt(s)
  s = 1
  return s

def add_arrow_head(real, imag, length, arrow_head_angle, colors_interp, ax=None):

  if ax is None:
    ax = plt.gca()
  
  fig = ax.get_figure()

  bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
  aspect_ratio = bbox.width / bbox.height
  a = 1/np.sqrt(aspect_ratio)

  if np.nanmin(real) < plt.xlim()[0]:
    #print("Out of bounds")
    idx = np.nanargmin(np.abs(real-plt.xlim()[0])) + 4
  else:
    idx = 0
  
  stop = np.array([[real[idx], imag[idx]]])
  color = colors_interp[idx]
  shaft_angle = np.arctan2((imag[idx+1]-imag[idx])/a, (real[idx+1]-real[idx])*a)

  hat = stop + length*np.array([
    [np.cos(shaft_angle+arrow_head_angle)/a, np.sin(shaft_angle+arrow_head_angle)*a], 
    [0,0], 
    [np.cos(shaft_angle-arrow_head_angle)/a, np.sin(shaft_angle-arrow_head_angle)*a]
  ])
  
  ax.plot(*hat.T, color=color, zorder=-1)

def plot_trajectories(npoles, scale_osc, scale_damping, domain, 
  plot_domain = [1-0.06j, 2.5+0.01j], fig_width = 90*mm, fig_height = 80*mm,
  unit = "eV", axs=None, cbar=True, labels=True, plot_neg_eps_r=False, d_limit=np.inf, 
  color_thickness=False, num_modes = 4, upsample=10, label_suffix="", 
  thickness_color_norm=None, eps_background=True, linewidth_modes=[], 
  arrows=False, arrow_head_angle=np.pi/16, arrow_head_length=0.003, 
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

  if eps_background:
    pcm = plt.pcolormesh(c_(E_r), c_(E_i), np.real(eps), norm=mpl.colors.SymLogNorm(linthresh=0.1,
                      vmin=-1000.0, vmax=1000.0, base=10), shading='gouraud',
                      cmap="RdBu", rasterized=True, zorder=-2)

    if cbar:
      cb = plt.colorbar(
        label="$\Re\{\\varepsilon_\mathrm{r}\}$", 
        ticks=[-1e3, -1, 0, 1, 1e3]
      )
      cb.set_ticklabels(["-$10^3$", -1, 0, 1, "$10^3$"])
  else:
    pcm = None

  plt.ylim(min(c_(e_i)), max(c_(e_i)))
  plt.xlim(min(c_(e_r)), max(c_(e_r)))

  ## Material Poles and Zeros
  mew =0.5
  plt.scatter(
    poles.real, poles.imag, 
    marker="x", color="k", linewidths=mew, zorder=10
  )
  plt.scatter(
    zeros.real, zeros.imag, 
    facecolors='none', edgecolors="k", linewidths=mew, zorder=10
  )

  # Interpolate thicknesses to make plot smoother
  interp_d = np.linspace(min(thickness), min(max(thickness), d_limit), (len(thickness)-1)*upsample+1) [::-1]
  cmap = mpl.cm.viridis_r
  if thickness_color_norm is None:
    thickness_color_norm = mpl.colors.Normalize(vmin=min(interp_d), vmax=max(interp_d))
  colors_interp = cmap(thickness_color_norm(interp_d))

  # Plot Trajectories
  for i, (pole, res) in enumerate(zip(poles_tracked.T, residues_tracked.T)):
    real = np.interp(interp_d, thickness, pole.real)
    imag = np.interp(interp_d, thickness, pole.imag)
    s = size(np.interp(interp_d, thickness, np.abs(res)), imag)
    plt.scatter(c_(real), c_(imag), c=colors_interp, edgecolor='none', s=s, rasterized=True)
    print(np.nanmax(np.abs(res)))
    if arrows and np.nanmax(np.abs(res))>5e-3:
      add_arrow_head(c_(real), c_(imag), arrow_head_length, arrow_head_angle, colors_interp)

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

  
  all_modes = np.array(range(len(poles_tracked.T)), dtype=int)
  for pole, res in zip(poles_tracked.T[all_modes], residues_tracked.T[all_modes]):
      real = np.interp(interp_d, thickness, pole.real)
      imag = np.interp(interp_d, thickness, pole.imag)
      s = size(np.interp(interp_d, thickness, np.abs(res)), imag)

      plt.scatter(c_(real), interp_d, color=[0.8,0.8,0.8], edgecolor='none', s=s, rasterized=True)

  all_modes = set(all_modes)
  for i in range(1, num_modes+1)[::-1]:
    selection, _ = select_modes(npoles, scale_osc, scale_damping, domain, modenumber=i)
    all_modes -= set(selection)

    for j, (pole, res) in enumerate(zip(poles_tracked.T[selection], residues_tracked.T[selection])):
      real = np.interp(interp_d, thickness, pole.real)
      imag = np.interp(interp_d, thickness, pole.imag)

      if i in linewidth_modes:
        plt.fill_betweenx(interp_d, 
          c_(real-imag), c_(real+imag), 
          color=[0.9,0.9,0.9],
          zorder=-2
        )
          
      s = size(np.interp(interp_d, thickness, np.abs(res)), imag)

      c0_rgba = mcolors.to_rgba(f"C{i-1}")
      c0_hsv = colorsys.rgb_to_hsv(*c0_rgba[:3])

      sat = (((npoles)-j)/(npoles) - 1)*0.8 + 1
      if i > 1:
        sat *= 0.5
      new_saturation = c0_hsv[1] * sat
      new_rgb = colorsys.hsv_to_rgb(c0_hsv[0], new_saturation, c0_hsv[2])
      new_rgba = (*new_rgb, c0_rgba[3])

      if color_thickness:
        plt.scatter(c_(real), interp_d, c=colors_interp, edgecolor='none', s=s, rasterized=True)
      else:
        plt.scatter(c_(real), interp_d, color=new_rgba, edgecolor='none', s=s, rasterized=True)

  plt.ylim(0, np.nanmax(interp_d))
  plt.xlim(min(c_(e_r)), max(c_(e_r)))

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
    fig.align_ylabels()

  return pcm, cmap, thickness_color_norm

# %%
if __name__ == "__main__":
    unit = "eV"
    # plt.figure()
    # plot_trajectories(1, 1, 1, domain, unit=unit, fig_width=90*mm, fig_height=50*mm)
    # plt.savefig("out/SinglePole.pdf", dpi=1200)

    # %%
    fig, axss = plt.subplots(
        3, 3, sharex="col",
        figsize=(90*mm,70*mm),  
        width_ratios=[1,0.03,0.03],
        height_ratios=[0.6, 1, 0.6], constrained_layout=True
    )
    
    e_r = np.linspace(domain[0].real, domain[1].real, 1200)
    eps_r = eps_surmof(e_r, 3, 1, 1)
    plt.sca(axss[0, 0])
    plt.plot(e_r, eps_r.real, color="gray", label=r"$\Re\{\varepsilon_\mathrm{r}\}$")
    plt.plot(e_r, eps_r.imag, color="k", label=r"$\Im\{\varepsilon_\mathrm{r}\}$")
    plt.legend(loc="upper right", fontsize=6, frameon=True)
    plt.ylabel(r"$\varepsilon_\mathrm{r}$")

    plt.fill_between(e_r, 0, 1, hatch="\\\\\\", where=eps_r.real<0, color="none", edgecolor="k", transform=plt.gca().get_xaxis_transform(), lw=0.5)

    pcm, cmap, norm = plot_trajectories(
      3, 1, 1, domain, unit=unit, 
      plot_neg_eps_r=True, axs=axss[1:,0], plot_domain=[1.5-0.06j, 2.1], 
      upsample=10, cbar=False, arrows=True
    )

    plt.xlim(1.5, 2.1)
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = plt.colorbar(mappable, cax=axss[1, 1], label=f"$d$ [{um}]")

    cb = plt.colorbar(pcm, cax=axss[1, 2], label="$\Re\{\\varepsilon_\mathrm{r}\}$", ticks=[-1e3, -1, 0, 1, 1e3])
    cb.set_ticklabels(["-$10^3$", -1, 0, 1, "$10^3$"])

    unused_axis = [[0,1], [0,2], [2,1], [2,2]]
    for ax in unused_axis:
      axss[ax[0], ax[1]].axis("off")


    for i, ax in enumerate(axss[:, 0]):
      letter = chr(ord("a")+i)
      ax.annotate(
            f" ({letter})",
            xy=(0, 1), xycoords='axes fraction',
            xytext=(+0.5, -0.5), textcoords='offset fontsize',
            fontsize='medium', verticalalignment='top', fontfamily='serif',
            bbox=dict(facecolor=(1,1,1,0.8), edgecolor='none', pad=2.0))

    plt.savefig("out/ThreePole.pdf", dpi=1200)

    # # %%
    # large_domain = [0.7-0.7j, 5.0+0.05j]
    # plt.figure()
    # plot_trajectories(1, 1, 1, large_domain, unit=unit, plot_neg_eps_r=True, 
    #                   fig_width=90*mm, fig_height=50*mm, cbar=False, arrows=True)
    # plt.xlim(1.5, 2.1)
    # plt.savefig("out/zero_stop.pdf")

    # %%
    # %matplotlib widget

    oscs = [1, 0.1, 0.05, 0.025]
    plot_domain2 = [1.6-0.149j, 1.82] #+0.05j]
    plot_domain1 = [1.3-0.149j, 2.2] #+0.05j]

    thickness_color_ranges = [[0.19, 0.235], [0,0.4]]
    thresh = 1

    pole_to_color = ["C0", "C1", "C2", "C0", "C3"] + ["none"]*20
    pole_to_color[11] = "C2"
    pole_to_color[5] = pole_to_color[10] = pole_to_color[16] = "C3"

    fig, axss = plt.subplots(
        2, len(oscs)+3, sharex="col",
        figsize=(180*mm,60*mm), 
        #constrained_layout=True, 
        width_ratios=[1.4,1,1,0.06, 0.08,1,0.06], #([1]*2 + [0.06])*2
        height_ratios=[1, 0.6]
        )

    suffix = 1

    norms = []
    #oscs = np.array(oscs)#[-2:]
    for osc, axs in list(zip(oscs[::-1], axss.T[np.array([0,1,2,5])])):

      vmin, vmax = thickness_color_ranges[osc >= thresh]
      pcm, cmap, norm = plot_trajectories(
          1, osc, 1, domain, 
          plot_domain=plot_domain1 if osc==1 else plot_domain2, 
          unit=unit, axs=axs, cbar=False, labels=False, d_limit=0.4, 
          upsample=10, label_suffix = suffix,
          thickness_color_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax),
          eps_background=False, linewidth_modes=[1], arrows=True, 
          arrow_head_angle=np.pi/16 if osc==1 else np.pi/10
      )
      suffix += 1
      axs[0].set_title(rf"$\eta = \num{{{osc:.3f}}}$")
      norms.append(norm)

    ## plot inset 1
    axins = axss[1,1].inset_axes(
      [0.55, 0.09, 0.4, 0.32],
      xlim=(1.73, 1.74),
      ylim=(0.205, 0.23),
      xticklabels=[], yticklabels=[]
    )

    npoles = 1
    scale_osc = 0.05
    scale_damping = 1
    poles, residues, thickness, material_poles = load_data(
      npoles, scale_osc, scale_damping, domain
    )
    poles_tracked, residues_tracked = track_qnms(poles, residues)
    selection, _ = select_modes(npoles, scale_osc, scale_damping, domain, modenumber=1)
    poles_tracked = poles_tracked.T[selection]
    axins.plot(poles_tracked.real.T, thickness, color="#96a8b4")
    axins.tick_params(axis='y', which='minor', left=False, right=False)
    
    # Draw the indicator or zoom lines.
    axss[1,1].indicate_inset_zoom(axins, edgecolor="black")
  
    ## plot inset 2
    axins = axss[0,0].inset_axes(
      [0.65, 0.45, 0.5, 0.33],
      xlim=(1.705, 1.713),
      ylim=(-0.0115, -0.0108),
      xticklabels=[], yticklabels=[]
    )
    axss[0,0].set_zorder(5)
    axins.set_zorder(6)

    npoles = 1
    scale_osc = 0.025
    scale_damping = 1
    poles, residues, thickness, material_poles = load_data(
      npoles, scale_osc, scale_damping, domain
    )
    poles_tracked, residues_tracked = track_qnms(poles, residues)
    mask = thickness < 10 #0.4
    poles_tracked = poles_tracked[mask]
    residues_tracked = residues_tracked[mask]
    thickness = thickness[mask]

    interp_d = np.linspace(min(thickness), max(thickness), (len(thickness)-1)*10+1) [::-1]
    cmap = mpl.cm.viridis_r
    thickness_color_norm = norms[0]
    colors_interp = cmap(thickness_color_norm(interp_d))

    for i, (pole, res) in enumerate(zip(poles_tracked.T, residues_tracked.T)):
      real = np.interp(interp_d, thickness, pole.real)
      imag = np.interp(interp_d, thickness, pole.imag)

      color = pole_to_color[i]
      axins.scatter(real, imag, edgecolor='none', s=1, c=color, rasterized=True)
      color = "gray" if color == "none" else color
      add_arrow_head(real, imag, 0.0001, np.pi/16, [color]*len(colors_interp), ax = axins)

      poi_labels = {3: "FP", 0: "M"}
      thickness_pts = [0.2, 0.209, 0.22]
      if i in poi_labels:
        real_labels = np.interp(thickness_pts, thickness, pole.real)
        imag_labels = np.interp(thickness_pts, thickness, pole.imag)
        axss[0,0].scatter(real_labels, imag_labels, marker="v", zorder=12, color="k")
        for j, (re, im) in enumerate(zip(real_labels, imag_labels)):
          if poi_labels[i] == "M":
            re += 0.01 * (j-1)
            im -= 0.02
            va = 'top'
          else:
            va = 'bottom'
          axss[0,0].annotate(f"{poi_labels[i]}{j+1}", (re, im+0.01), fontsize=6, va=va , ha='center')

    e_r = np.linspace(1.7, 1.72, 51)
    e_i = np.linspace(-0.013, -0.009, 31)
    E_r, E_i = np.meshgrid(e_r, e_i)
    E = E_r +1j*E_i
    
    eps = eps_surmof(E, npoles, scale_osc, scale_damping)
    ds = 1
    _, _, _, poles = diffaaable.aaa(E[::ds, ::ds], eps[::ds, ::ds], tol=1e-8)
    _, _, _, zeros = diffaaable.aaa(E[::ds, ::ds], 1/eps[::ds, ::ds], tol=1e-8)

    mew = 0.5
    axins.scatter(
      poles.real, poles.imag, 
      marker="x", color="k", linewidths=mew
    )
    axins.scatter(
      zeros.real, zeros.imag, 
      facecolors='none', edgecolors="k", linewidths=mew
    )

    # Draw the indicator or zoom lines.
    axss[0,0].indicate_inset_zoom(axins, edgecolor="black")

    ## colorbars etc.
    if len(norms) < 4:
      norms = [norms[0]]*4
    axss[1,0].set_ylim(0, 0.4)

    shrink=0.8
    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norms[1])
    cb = fig.colorbar(mappable, cax=axss[0, 3], shrink=shrink)
    #cb.set_label(f"$d$ [{um}]", labelpad=-10)
    cb.ax.set_title(f"$d$ [{um}]", loc='left')
    cb.ax.zorder = -1

    mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norms[3])
    cb = fig.colorbar(mappable, cax=axss[0, -1], shrink=shrink)
    cb.ax.set_title(f"$d$ [{um}]", loc='left')
    cb.ax.zorder = -1

    #axss[0,0].set_title(f"$\eta$ = {oscs[-1]}")
    fig.supxlabel(f"$\Re\{{{qty_str[unit]}\}}$ [{unit}]", y=0.08)
    axss[0,0].set_ylabel(f"$\Im\{{{qty_str[unit]}\}}$ [{unit}]")
    axss[1,0].set_ylabel(f"$d$ [{um}]")

    fig.align_ylabels()

    for axs in axss.T[np.array([1,2,5])]:
      axs[0].sharey(axss[0,0])
      axs[0].tick_params(axis='y',labelleft=False)
      axs[1].sharey(axss[1,0])
      axs[1].tick_params(axis='y',labelleft=False)

    axss[1, 3].axis("off")
    axss[1, 6].axis("off")

    axss[0, 4].axis("off")
    axss[1, 4].axis("off")


    # zorder_indicators = 0
    # axesB = [
    #   [axss[0, 0], axss[0, 2]],
    #   [axss[0, 4], axss[0, 4]],
    # ]
    # for i_cbar in range(2):
    #   for corner in range(2):
    #     con = ConnectionPatch(
    #       (corner, thickness_color_ranges[i_cbar][corner]),
    #       (corner, corner),
    #       coordsA='data', coordsB='axes fraction', axesA=axss[1, -1 if i_cbar else 3], axesB=axesB[i_cbar][corner],
    #       zorder=zorder_indicators)
    #     fig.add_artist(con)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.1)

    print("Start Rendering")
    plt.savefig("out/OscReduction.pdf", dpi=1200)

    # %%
    # poles, residues, thickness, material_poles = load_data(
    #   1, 1, 1, domain
    # )
    # axss[0,0].axvline(material_poles[0].real)

    # %%
    plt.figure()
    npoles = 0
    scale_osc = 1
    scale_damping = 1
    poles, residues, thickness, material_poles = load_data(
      npoles, scale_osc, scale_damping, domain
    )
    poles_tracked, residues_tracked = track_qnms(poles, residues)
    mask = thickness < 0.4
    poles_tracked = poles_tracked[mask]
    residues_tracked = residues_tracked[mask]
    thickness = thickness[mask]

    pole = poles_tracked.T[0]

    for ax in axss[0][np.array([0,1,2,5])]:
      plt.sca(ax)
      plt.plot(pole.real, pole.imag, "--", color="grey")

    plt.savefig("out/OscReduction_with_cav.pdf", dpi=1200)

    # %%
